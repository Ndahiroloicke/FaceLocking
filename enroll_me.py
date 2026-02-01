from src.embed import ArcFaceEmbedder
from src.enroll import enroll_identity

embedder = ArcFaceEmbedder("models/arcface.onnx")

image_paths = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]

count, folder = enroll_identity(
    name="YourName",  # Change this
    image_paths=image_paths,
    embedder=embedder,
    output_dir="data/identities",
)

print(f"✓ Enrolled {count} samples")