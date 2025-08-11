import argparse, os, sys, torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

def is_clip(name:str)->bool:
    return name.lower() in ["clip-vit-l-14","clip_vit_l_14","clip-vit-l/14","clip"]

def is_convnext(name:str)->bool:
    return name.lower() in ["convnext_base_in22k","convnext-base-in22k","convnext_b"]

def build_model_and_transform(model_name:str, input_size:int=224):
    if is_convnext(model_name):
        import timm
        m = timm.create_model("convnext_base_in22k", pretrained=False, num_classes=2)
        # timm의 변환을 사용 (ImageNet mean/std, center crop)
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        cfg = resolve_data_config({}, model=m)
        tfm = create_transform(input_size=input_size, is_training=False, **cfg)
        return m, tfm
    elif is_clip(model_name):
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=None,  # 외부 pretrained 불러오지 않음 (우린 .pth 로드)
        )
        # 분류 헤드 붙이기 (DRCT 예시에서 embedding_size=1024를 많이 씀)
        class CLIPHead(nn.Module):
            def __init__(self, clip_model, num_classes=2):
                super().__init__()
                self.clip = clip_model
                # ViT-L/14의 visual.embed_dim은 1024
                self.head = nn.Linear(self.clip.visual.embed_dim, num_classes)

            def forward(self, x):
                # open_clip 의 encode_image는 정규화 처리까지 포함
                with torch.no_grad():
                    feats = self.clip.encode_image(x)  # [B, 1024]
                logits = self.head(feats)
                return logits
        wrapped = CLIPHead(model)
        return wrapped, preprocess
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def load_weights(model:nn.Module, weights_path:str, map_location="cpu"):
    sd = torch.load(weights_path, map_location=map_location)
    # pth가 state_dict 그 자체이거나 {"state_dict": ...} 인 경우 모두 처리
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)

@torch.no_grad()
def infer_images(model, transform, image_paths, device="cpu", class_names=("real","fake")):
    model.eval().to(device)
    results = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.tensor(probs).argmax().item())
        pred_name = class_names[pred_idx]
        results.append((p, pred_name, probs))
        print(f"{p} -> {pred_name}  (real={probs[0]:.4f}, fake={probs[1]:.4f})")
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="convnext_base_in22k | clip-ViT-L-14")
    ap.add_argument("--weights_path", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="image paths")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--input_size", type=int, default=224)
    args = ap.parse_args()

    model, transform = build_model_and_transform(args.model_name, args.input_size)
    load_weights(model, args.weights_path, map_location="cpu")
    infer_images(model, transform, args.images, device=args.device)

if __name__ == "__main__":
    main()
