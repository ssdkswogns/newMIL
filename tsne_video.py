import argparse
import glob
import os
from typing import Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import imageio.v2 as imageio


def load_epoch_files(input_dir: str) -> Tuple[list, list]:
    """
    Returns sorted list of (epoch, raw_path, tsne_path_or_none)
    """
    raw_files = sorted(glob.glob(os.path.join(input_dir, "epoch_*_raw.npz")))
    epochs = []
    tsne_files = []
    for raw_path in raw_files:
        base = raw_path.replace("_raw.npz", "")
        tsne_path = base + "_tsne.npz"
        ep_str = os.path.basename(base).split("_")[1]
        try:
            epoch = int(ep_str)
        except Exception:
            epoch = ep_str
        epochs.append((epoch, raw_path, tsne_path if os.path.exists(tsne_path) else None))
    return epochs, raw_files


def compute_tsne(data: np.ndarray, perplexity: float, n_iter: int, random_state: int = 0):
    if data.shape[0] <= 2:
        return None
    perp = min(perplexity, max(1, data.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        n_iter=n_iter,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    return tsne.fit_transform(data)


def assign_class_with_prototypes(
    inst_embed: np.ndarray,
    inst_pred: np.ndarray,
    proto_embed: Optional[np.ndarray],
    proto_label: Optional[np.ndarray],
    sim_thresh: float = 0.5,
) -> np.ndarray:
    """
    Prototype-instance pairing aligned to training:
    - Use prototypes if available: nearest prototype by cosine; keep only if similarity >= sim_thresh, else -1.
    - If prototypes are missing, fall back to inst_pred (argmax saved from training).
    - If nothing is available, return -1.
    """
    N = inst_embed.shape[0]

    if proto_embed is not None and proto_embed.size > 0:
        # filter out zero-norm prototypes to avoid collapsing to one point
        proto_norm_val = np.linalg.norm(proto_embed, axis=1, keepdims=True)
        valid_proto = (proto_norm_val.squeeze(-1) > 1e-8)
        if valid_proto.sum() == 0:
            proto_embed = None
        else:
            proto_embed = proto_embed[valid_proto]
            proto_label = proto_label[valid_proto]

    if proto_embed is not None and proto_embed.size > 0:
        inst_norm = inst_embed / (np.linalg.norm(inst_embed, axis=1, keepdims=True) + 1e-8)
        proto_norm = proto_embed / (np.linalg.norm(proto_embed, axis=1, keepdims=True) + 1e-8)
        sim = inst_norm @ proto_norm.T  # [N, P]
        nearest = np.argmax(sim, axis=1)
        max_sim = sim[np.arange(sim.shape[0]), nearest]
        assigned = np.where(max_sim >= sim_thresh, proto_label[nearest], -1)
        return assigned.astype(int)

    if inst_pred is not None and inst_pred.size > 0:
        return inst_pred.astype(int)

    return -1 * np.ones(N, dtype=int)


def open_writer_safe(output_path: str, fps: int):
    """
    Try to open requested writer; if backend missing for mp4, raise with guidance.
    """
    try:
        writer = imageio.get_writer(output_path, fps=fps)
        return writer, output_path
    except ValueError as e:
        msg = (
            f"Could not open {output_path} for writing ({e}). "
            "Install an ffmpeg-capable backend, e.g., `pip install imageio[ffmpeg]`, "
            "or provide a path with a supported extension."
        )
        raise ValueError(msg)


def collect_all_classes(epochs: list) -> np.ndarray:
    cls_ids = set()
    for ep, raw_path, tsne_path in epochs:
        raw = np.load(raw_path)
        inst_pred = raw.get("inst_pred", np.array([]))
        proto_label = raw.get("proto_label", np.array([]))
        if inst_pred is not None:
            cls_ids.update(np.array(inst_pred).flatten().tolist())
        if proto_label is not None:
            cls_ids.update(np.array(proto_label).flatten().tolist())
        if tsne_path is not None and os.path.exists(tsne_path):
            tsne = np.load(tsne_path)
            if "label" in tsne:
                cls_ids.update(np.array(tsne["label"]).flatten().tolist())
    cls_ids = [int(c) for c in cls_ids if np.isfinite(c)]
    cls_ids = sorted(set(cls_ids))
    return np.array(cls_ids, dtype=int)


def build_color_map(all_classes: np.ndarray) -> Dict[int, tuple]:
    cmap = plt.get_cmap("tab20")
    color_map = {}
    for i, c in enumerate(all_classes):
        color_map[int(c)] = cmap(i % 20)
    # keep -1 gray
    color_map[-1] = (0.6, 0.6, 0.6, 0.5)
    return color_map


def render_frame(
    coords_inst: np.ndarray,
    label_inst: np.ndarray,
    coords_proto: Optional[np.ndarray],
    label_proto: Optional[np.ndarray],
    epoch: str,
    fig_size: Tuple[int, int] = (6, 6),
    color_map: Optional[Dict[int, tuple]] = None,
):
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")
    ax.text(0.02, 0.98, f"Epoch: {epoch}", transform=ax.transAxes, fontsize=12, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    default_color = color_map.get(-1, (0.6, 0.6, 0.6, 0.5)) if color_map else (0.6, 0.6, 0.6, 0.5)

    if coords_inst is not None and coords_inst.shape[0] > 0:
        inst_colors = [color_map.get(int(c), default_color) for c in label_inst] if color_map else default_color
        ax.scatter(coords_inst[:, 0], coords_inst[:, 1], s=6, c=inst_colors, alpha=0.7, label="instance")

    if coords_proto is not None and coords_proto.size > 0:
        # debug: track how many prototypes are being plotted
        print(f"[render_frame] epoch {epoch} prototypes: {coords_proto.shape[0]}, labels={label_proto.tolist() if hasattr(label_proto, 'tolist') else label_proto}")
        print(coords_proto)
        proto_colors = [color_map.get(int(c), default_color) for c in label_proto] if color_map else default_color
        ax.scatter(
            coords_proto[:, 0],
            coords_proto[:, 1],
            s=80,
            c=proto_colors,
            marker="*",
            edgecolors="k",
            linewidths=0.8,
            label="prototype",
        )

    # build legend with fixed class-color mapping
    handles = []
    labels = []
    if color_map:
        from matplotlib.lines import Line2D
        for cls_id, clr in sorted(color_map.items(), key=lambda kv: kv[0]):
            if cls_id == -1:
                continue
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=clr, markersize=8, label=str(cls_id)))
            labels.append(str(cls_id))
    if handles:
        ax.legend(handles=handles, title="class", loc="lower right", framealpha=0.7)

    plt.tight_layout()

    # Convert to RGB array
    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def main():
    parser = argparse.ArgumentParser(description="Create t-SNE video from saved epoch npz files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing epoch_*_raw.npz (and optional _tsne.npz).")
    parser.add_argument("--output", type=str, default="tsne_epochs.mp4", help="Output video path.")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for video.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity if coords need to be computed.")
    parser.add_argument("--n_iter", type=int, default=500, help="t-SNE iterations if coords need to be computed.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(6, 6), help="Figure size for each frame.")
    parser.add_argument("--max_frames", type=int, default=0, help="Limit number of epochs to render (0 means all).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for t-SNE.")
    parser.add_argument("--num_classes", type=int, default=-1, help="If set >0, fix color map to [0..num_classes-1].")
    parser.add_argument("--color_sim_thresh", type=float, default=0.5, help="Cosine similarity threshold for prototype-instance mapping.")
    parser.add_argument("--force_tsne", action="store_true", help="Ignore saved _tsne.npz and recompute t-SNE.")
    parser.add_argument("--proto_scale", type=float, default=1.0, help="Scale factor applied to prototype embeddings before joint t-SNE.")
    parser.add_argument("--proto_repeat", type=int, default=1, help="Repeat prototypes to increase their weight in joint t-SNE.")
    parser.add_argument("--sim_pull_thresh", type=float, default=0.6, help="If >0, create pull anchors for instances with cosine sim to nearest proto above this threshold.")
    parser.add_argument("--sim_pull_weight", type=float, default=0.5, help="Interpolation weight for pull anchors: anchor = (1-w)*inst + w*proto.")
    parser.add_argument("--sim_pull_repeat", type=int, default=1, help="Repeat pull anchors to strengthen pull effect.")
    args = parser.parse_args()

    epochs, _ = load_epoch_files(args.input_dir)
    if len(epochs) == 0:
        raise FileNotFoundError(f"No epoch_*_raw.npz found in {args.input_dir}")

    if args.max_frames > 0:
        epochs = epochs[:args.max_frames]

    if args.num_classes and args.num_classes > 0:
        all_classes = np.arange(args.num_classes, dtype=int)
    else:
        all_classes = collect_all_classes(epochs)
    color_map = build_color_map(all_classes)

    writer, actual_output = open_writer_safe(args.output, args.fps)

    for (ep, raw_path, tsne_path) in epochs:
        raw = np.load(raw_path)
        inst_embed = raw.get("inst_embed", np.array([]))
        inst_pred = raw.get("inst_pred", np.array([]))
        proto_embed = raw.get("proto_embed", np.array([]))
        proto_label = raw.get("proto_label", np.array([]))

        # Determine coords
        coords_inst = None
        coords_proto = None

        if tsne_path is not None and not args.force_tsne:
            tsne = np.load(tsne_path)
            coords = tsne["coords"]
            labels = tsne["label"] if "label" in tsne else None
            kinds = tsne["kind"] if "kind" in tsne else None
            if kinds is not None:
                inst_mask = kinds == 0
                proto_mask = kinds == 1
                coords_inst = coords[inst_mask]
                coords_proto = coords[proto_mask]
                if labels is not None:
                    if inst_pred is None or inst_pred.size == 0:
                        inst_pred = labels[inst_mask]
                    if (proto_label is None or proto_label.size == 0) and proto_mask.any():
                        proto_label = labels[proto_mask]
            else:
                # fallback: assume first part is instance, tail is proto based on counts
                n_inst = inst_embed.shape[0]
                coords_inst = coords[:n_inst]
                coords_proto = coords[n_inst:]
        else:
            # need to run tsne on combined
            combined = inst_embed
            pull_points = None
            if proto_embed is not None and proto_embed.size > 0:
                # drop zero prototypes before combining
                proto_norm = np.linalg.norm(proto_embed, axis=1)
                mask = proto_norm > 1e-8
                proto_embed = proto_embed[mask]
                if proto_label is not None and proto_label.size > 0:
                    proto_label = proto_label[mask]
                if proto_embed.size > 0 and args.proto_scale != 1.0:
                    proto_embed = proto_embed * args.proto_scale
                if proto_embed.size > 0 and args.proto_repeat > 1:
                    proto_embed = np.repeat(proto_embed, args.proto_repeat, axis=0)
                    if proto_label is not None and proto_label.size > 0:
                        proto_label = np.repeat(proto_label, args.proto_repeat, axis=0)

                # optional pull anchors: instances close to a prototype
                if proto_embed.size > 0 and args.sim_pull_thresh > 0:
                    inst_norm = inst_embed / (np.linalg.norm(inst_embed, axis=1, keepdims=True) + 1e-8)
                    proto_normed = proto_embed / (np.linalg.norm(proto_embed, axis=1, keepdims=True) + 1e-8)
                    sim = inst_norm @ proto_normed.T
                    nearest = np.argmax(sim, axis=1)
                    max_sim = sim[np.arange(sim.shape[0]), nearest]
                    sel = max_sim >= args.sim_pull_thresh
                    if sel.any():
                        proto_near = proto_embed[nearest[sel]]
                        inst_sel = inst_embed[sel]
                        w = np.clip(args.sim_pull_weight, 0.0, 1.0)
                        pull_points = (1 - w) * inst_sel + w * proto_near
                        if args.sim_pull_repeat > 1:
                            pull_points = np.repeat(pull_points, args.sim_pull_repeat, axis=0)

                if proto_embed.size > 0:
                    combined = np.concatenate([inst_embed, proto_embed], axis=0)
                if pull_points is not None and pull_points.size > 0:
                    combined = np.concatenate([combined, pull_points], axis=0)

            coords_all = compute_tsne(combined, args.perplexity, args.n_iter, random_state=args.seed)
            if coords_all is None:
                coords_inst = np.zeros((inst_embed.shape[0], 2))
                coords_proto = np.zeros((proto_embed.shape[0], 2)) if proto_embed is not None else None
            else:
                n_inst = inst_embed.shape[0]
                n_proto = proto_embed.shape[0] if proto_embed is not None else 0
                coords_inst = coords_all[:n_inst]
                coords_proto = coords_all[n_inst:n_inst + n_proto] if n_proto > 0 else None

        # Assign labels
        inst_labels = assign_class_with_prototypes(
            inst_embed,
            inst_pred,
            proto_embed,
            proto_label,
            sim_thresh=args.color_sim_thresh,
        )

        frame = render_frame(
            coords_inst=coords_inst,
            label_inst=inst_labels,
            coords_proto=coords_proto,
            label_proto=proto_label if proto_label is not None and proto_label.size > 0 else None,
            epoch=str(ep),
            fig_size=tuple(args.figsize),
            color_map=color_map,
        )
        writer.append_data(frame)
        print(f"[Video] added epoch {ep}")

    writer.close()
    print(f"Saved video to {actual_output}")


if __name__ == "__main__":
    main()
