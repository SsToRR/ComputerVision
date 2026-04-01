import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

from facenet_pytorch import MTCNN, InceptionResnetV1

# =========================
# CONFIG
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = Path(__file__).resolve().parent / "face_lab_dataset"

MIN_IMAGES_PER_ID = 3
MAX_IDENTITIES = 20
MAX_IMAGES_PER_ID = 40
TEST_SIZE = 0.3

# =========================
# LOAD DATASET
# =========================
# Loads face images from per-person folders and returns labeled samples.
def load_samples_from_folder(root_dir, min_images_per_id, max_identities, max_images_per_id):
    counts = []
    for person_dir in root_dir.iterdir():
        if not person_dir.is_dir():
            continue
        image_paths = sorted(
            path for path in person_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if len(image_paths) >= min_images_per_id:
            counts.append((person_dir.name, image_paths))

    counts.sort(key=lambda item: len(item[1]), reverse=True)
    selected = counts[:max_identities]

    samples = []
    for label, image_paths in selected:
        for image_path in image_paths[:max_images_per_id]:
            image = Image.open(image_path).convert("RGB")
            samples.append({
                "image": np.asarray(image, dtype=np.uint8),
                "label": label,
            })

    return samples


if DATASET_DIR.exists():
    print(f"Loading dataset from folder: {DATASET_DIR}")
    samples = load_samples_from_folder(
        DATASET_DIR,
        min_images_per_id=MIN_IMAGES_PER_ID,
        max_identities=MAX_IDENTITIES,
        max_images_per_id=MAX_IMAGES_PER_ID,
    )
else:
    print("Local folder dataset not found, falling back to sklearn LFW.")
    lfw = fetch_lfw_people(color=True, resize=1.0, min_faces_per_person=MIN_IMAGES_PER_ID)

    images = lfw.images
    targets = lfw.target
    names = lfw.target_names

    counts = Counter(targets)
    selected_ids = [tid for tid, _ in counts.most_common(MAX_IDENTITIES)]

    samples = []
    for tid in selected_ids:
        idxs = np.where(targets == tid)[0][:MAX_IMAGES_PER_ID]
        for idx in idxs:
            samples.append({
                "image": images[idx].astype(np.uint8),
                "label": names[tid]
            })

print("Total samples:", len(samples))
print("Identities:", len(set([s["label"] for s in samples])))


# Prints dataset statistics and shows class distribution plots.
def analyze_dataset(data):
    counts = Counter(sample["label"] for sample in data)
    ordered = counts.most_common()

    print("\n=== DATASET ANALYSIS ===")
    print("Number of identities:", len(counts))
    print("Total images:", len(data))
    print("Min images per identity:", min(counts.values()))
    print("Max images per identity:", max(counts.values()))
    print("Average images per identity:", np.mean(list(counts.values())))
    print("Top identities by image count:")
    for label, count in ordered[:10]:
        print(f"{label}: {count}")

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(ordered)), [count for _, count in ordered])
    plt.xlabel("Identity index (sorted by count)")
    plt.ylabel("Number of images")
    plt.title("Images per Identity")
    plt.tight_layout()
    plt.show()

    print("Variation summary:")
    print("The dataset contains unconstrained face photos with changes in pose, lighting, expression, background, and image quality.")


analyze_dataset(samples)

# =========================
# TRAIN / TEST SPLIT
# =========================
labels = [s["label"] for s in samples]
idxs = np.arange(len(samples))

train_idx, test_idx = train_test_split(
    idxs,
    test_size=TEST_SIZE,
    stratify=labels,
    random_state=SEED
)

train_samples = [samples[i] for i in train_idx]
test_samples  = [samples[i] for i in test_idx]

# =========================
# FACE DETECTION (MTCNN)
# =========================
mtcnn = MTCNN(image_size=160, margin=14, device=DEVICE)

# Detects, aligns, and normalizes a face crop from an input image.
def detect_align(img):
    aligned = mtcnn(Image.fromarray(img))
    if aligned is None:
        # fallback crop
        h, w = img.shape[:2]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        cropped = img[top:top + side, left:left + side]
        pil_img = Image.fromarray(cropped).resize((160, 160))
        return np.asarray(pil_img, dtype=np.uint8)
    else:
        img = aligned

    img = img.permute(1,2,0).cpu().numpy()
    img = ((img + 1) / 2 * 255).clip(0,255).astype(np.uint8)
    return img

# Applies face alignment to every sample in a dataset split.
def process(data):
    out = []
    for s in data:
        aligned = detect_align(s["image"])
        out.append({"image": aligned, "label": s["label"]})
    return out

print("Processing faces...")
train_data = process(train_samples)
test_data  = process(test_samples)

# =========================
# PREPARE ARRAYS
# =========================
labels_all = sorted(set([x["label"] for x in train_data]))
label_to_id = {l:i for i,l in enumerate(labels_all)}

X_train = np.stack([x["image"] for x in train_data])
y_train = np.array([label_to_id[x["label"]] for x in train_data])

X_test = np.stack([x["image"] for x in test_data])
y_test = np.array([label_to_id[x["label"]] for x in test_data])

# =========================
# FEATURE EXTRACTION (FaceNet)
# =========================
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Converts image arrays to FaceNet input tensor format.
def preprocess(x):
    x = torch.from_numpy(x).float()/255.0
    x = x.permute(0,3,1,2)
    x = (x - 0.5)/0.5
    return x

@torch.no_grad()
# Extracts normalized FaceNet embeddings in batches.
def extract(x):
    feats = []
    for i in range(0, len(x), 32):
        batch = preprocess(x[i:i+32]).to(DEVICE)
        emb = model(batch)
        emb = F.normalize(emb, dim=1)
        feats.append(emb.cpu().numpy())
    return np.vstack(feats)

print("Extracting embeddings...")
start = time.time()
train_emb = extract(X_train)
test_emb  = extract(X_test)
feature_extraction_time = time.time() - start

# =========================
# IDENTIFICATION
# =========================
# Builds one normalized prototype embedding per identity.
def build_prototypes(emb, labels):
    protos = []
    proto_labels = []
    for l in sorted(np.unique(labels)):
        p = emb[labels == l].mean(axis=0)
        p = p / np.linalg.norm(p)
        protos.append(p)
        proto_labels.append(l)
    return np.vstack(protos), np.array(proto_labels)

proto_emb, proto_labels = build_prototypes(train_emb, y_train)

sim = cosine_similarity(test_emb, proto_emb)

top1 = proto_labels[np.argmax(sim, axis=1)]
top3 = np.argsort(-sim, axis=1)[:, :3]

rank1 = (top1 == y_test).mean()
rank3 = np.mean([y_test[i] in proto_labels[top3[i]] for i in range(len(y_test))])

# =========================
# METRICS
# =========================
acc = accuracy_score(y_test, top1)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test,
    top1,
    average="macro",
    zero_division=0,
)

print("\n=== IDENTIFICATION ===")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("Rank-1:", rank1)
print("Rank-3:", rank3)


# Compares same-person and different-person embedding similarities.
def analyze_embedding_similarity(embeddings, labels):
    same_scores = []
    diff_scores = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = float(np.dot(embeddings[i], embeddings[j]))
            if labels[i] == labels[j]:
                same_scores.append(score)
            else:
                diff_scores.append(score)

    same_scores = np.array(same_scores)
    diff_scores = np.array(diff_scores)

    print("\n=== EMBEDDING ANALYSIS ===")
    print("Average same-identity similarity:", same_scores.mean())
    print("Average different-identity similarity:", diff_scores.mean())
    print("Same-identity similarity std:", same_scores.std())
    print("Different-identity similarity std:", diff_scores.std())

    plt.figure(figsize=(8, 4))
    plt.hist(same_scores, bins=30, alpha=0.6, label="Same identity")
    plt.hist(diff_scores, bins=30, alpha=0.6, label="Different identity")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequency")
    plt.title("Embedding Similarity Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


analyze_embedding_similarity(test_emb, y_test)


# Evaluates a classical PCA + 1NN baseline for face identification.
def compare_with_pca_baseline(X_train, y_train, X_test, y_test, top_k=3):
    X_train_flat = X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0
    X_test_flat = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0

    n_components = min(100, len(X_train_flat) - 1, X_train_flat.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)

    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn.fit(X_train_pca, y_train)
    pred = knn.predict(X_test_pca)

    distances, neighbors = knn.kneighbors(X_test_pca, n_neighbors=min(top_k, len(X_train_pca)))
    neighbor_labels = y_train[neighbors]
    rankk = np.mean([y_test[i] in neighbor_labels[i] for i in range(len(y_test))])

    acc = accuracy_score(y_test, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test,
        pred,
        average="macro",
        zero_division=0,
    )

    print("\n=== BASELINE COMPARISON: PCA + 1NN ===")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print(f"Rank-{min(top_k, len(X_train_pca))}:", rankk)
    print("Conclusion: FaceNet embeddings should outperform this classical PCA baseline on unconstrained face images.")


compare_with_pca_baseline(X_train, y_train, X_test, y_test)

# =========================
# VERIFICATION
# =========================
# Creates positive and negative image pairs for verification testing.
def create_pairs(labels, positive_pairs_per_identity=20, negative_pairs=500):
    pairs = []
    label_map = defaultdict(list)
    for i,l in enumerate(labels):
        label_map[l].append(i)

    # same
    for l in label_map:
        idxs = label_map[l]
        if len(idxs) >= 2:
            for _ in range(positive_pairs_per_identity):
                i,j = np.random.choice(idxs, 2, replace=False)
                pairs.append((i,j,1))

    # different
    labs = list(label_map.keys())
    for _ in range(negative_pairs):
        l1,l2 = np.random.choice(labs, 2, replace=False)
        i = np.random.choice(label_map[l1])
        j = np.random.choice(label_map[l2])
        pairs.append((i,j,0))

    return pairs

pairs = create_pairs(y_test)

scores = []
targets = []

for i,j,t in pairs:
    s = np.dot(test_emb[i], test_emb[j])
    scores.append(s)
    targets.append(t)

scores = np.array(scores)
targets = np.array(targets)

# threshold search
best_acc = 0
best_thr = 0

for thr in np.linspace(scores.min(), scores.max(), 200):
    pred = (scores >= thr).astype(int)
    acc_thr = (pred == targets).mean()
    if acc_thr > best_acc:
        best_acc = acc_thr
        best_thr = thr

# FAR / FRR
pred = (scores >= best_thr).astype(int)
far = np.mean(pred[targets==0] == 1)
frr = np.mean(pred[targets==1] == 0)

print("\n=== VERIFICATION ===")
print("Best threshold:", best_thr)
print("Accuracy:", best_acc)
print("FAR:", far)
print("FRR:", frr)

# =========================
# SIMPLE VISUALIZATION
# =========================
# Displays test images with ground-truth and predicted identities.
def show_predictions(indices, title_prefix):
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    for i in indices:
        plt.imshow(X_test[i])
        plt.title(f"{title_prefix} | GT: {id_to_label[y_test[i]]} | Pred: {id_to_label[top1[i]]}")
        plt.axis("off")
        plt.show()


correct_idx = np.where(top1 == y_test)[0]
wrong_idx = np.where(top1 != y_test)[0]

print("\n=== RECOGNITION EXAMPLES ===")
print("Correct predictions:", len(correct_idx))
print("Incorrect predictions:", len(wrong_idx))

show_predictions(correct_idx[:5], "Correct")
show_predictions(wrong_idx[:5], "Incorrect")


# Prints common misclassification cases and their similarity margins.
def error_analysis(similarity_matrix, y_true, y_pred, proto_labels, max_examples=5):
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    print("\n=== ERROR ANALYSIS ===")
    if len(wrong_idx) == 0:
        print("No misclassified examples on the current test split.")
        return

    for idx in wrong_idx[:max_examples]:
        predicted_label = proto_labels[np.argmax(similarity_matrix[idx])]
        best_score = float(np.max(similarity_matrix[idx]))
        second_best = np.partition(similarity_matrix[idx], -2)[-2] if similarity_matrix.shape[1] > 1 else best_score
        margin = float(best_score - second_best)

        print(
            f"Sample {idx}: GT={id_to_label[y_true[idx]]}, "
            f"Pred={id_to_label[y_pred[idx]]}, "
            f"best_sim={best_score:.4f}, margin={margin:.4f}"
        )

    print("Typical failure reasons:")
    print("1. Similar-looking identities produce close embeddings.")
    print("2. Side pose, occlusion, or poor crop quality reduces embedding quality.")
    print("3. Small training sets per identity make class prototypes less stable.")


error_analysis(sim, y_test, top1, proto_labels)


# Measures embedding extraction and single-query recognition time.
def measure_efficiency():
    print("\n=== COMPUTATIONAL EFFICIENCY ===")
    print("Total feature extraction time (train + test):", feature_extraction_time, "sec")
    print("Average feature extraction time per image:", feature_extraction_time / (len(X_train) + len(X_test)), "sec")

    start = time.time()
    _ = cosine_similarity(test_emb[:1], proto_emb)
    recognition_time = time.time() - start
    print("Recognition time for one query:", recognition_time, "sec")


measure_efficiency()
