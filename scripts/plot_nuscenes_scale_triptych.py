"""Render a three-task scale comparison from an explicit NPZ bundle.

Without --bundle, render a layout preview with NO fabricated results.
See docs/nuscenes-scale-figures.md for the input contract.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


COLORS = ['#374151', '#d97706', '#078477']
LABELS = ['Ground truth', 'Before scale recovery', 'After scale recovery']


def placeholder(ax, title, description):
    ax.set_facecolor('#f4f6f8')
    ax.set_title(title, fontsize=12, pad=10)
    ax.text(.5, .5, description, ha='center', va='center',
            transform=ax.transAxes, color='#64748b', fontsize=11, linespacing=1.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#d9dfe6')


def save(fig, out, name):
    fig.savefig(out / (name + '.png'), dpi=220, facecolor='white')
    fig.savefig(out / (name + '.pdf'), facecolor='white')
    plt.close(fig)


def preview(out):
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
    fig.suptitle('nuScenes | Metric scale recovery\nLAYOUT ONLY — data and predictions pending', fontsize=19)
    panels = [
        ('RGB input', 'Same camera / same frame\nVehicle or static-object ROI'),
        ('LiDAR reference', 'Sparse camera-z depth\nShared color range in meters'),
        ('Before recovery', 'Depth + object point cloud\nMeasured endpoint distance'),
        ('After recovery', 'Depth + object point cloud\nSame endpoints and viewpoint'),
        ('Input sequence', 'Selected RGB keyframes\nScene / camera / timestamps'),
        ('LiDAR reconstruction', 'Static scene in metric coordinates\nShared view + 10 m grid'),
        ('Before recovery', 'Fused prediction\nShared metric extent'),
        ('After recovery', 'Fused prediction\nShared metric extent'),
        ('Ego trajectory', 'GT / before / after overlay\nEqual-aspect meter axes'),
        ('Translation scale', 'Segment length / GT length\nIdeal ratio = 1'),
        ('Position error', 'Error versus traveled distance\nRigid alignment only'),
        ('Reporting', 'Path length and ATE in meters\nNo Sim(3) scale alignment'),
    ]
    for ax, (title, desc) in zip(axes.flat, panels):
        placeholder(ax, title, desc)
    save(fig, out, 'nuscenes_layout_preview')


def render(bundle, out, title, depth_max, segment):
    with np.load(bundle, allow_pickle=False) as source:
        d = dict(source)
    suffixes = ['gt', 'before', 'after']
    depths = [d['depth_' + s] for s in suffixes]
    rgb = d['rgb']
    if any(x.shape != rgb.shape[:2] for x in depths):
        raise ValueError('RGB and all depth maps must share H,W')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    fig.suptitle(title + ' | Depth estimation', fontsize=18)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB input')
    axes[0, 0].axis('off')
    placeholder(axes[1, 0], 'Metric measurement', 'Identical physical endpoints\n3D Euclidean distance\nCamera-z depth in meters')
    clouds = [d['object_' + s] for s in suffixes]
    allp = np.concatenate(clouds)
    bounds = [(np.nanmin(allp[:, k]), np.nanmax(allp[:, k])) for k in [0, 2]]
    for i, s in enumerate(suffixes):
        ax = axes[0, i + 1]
        valid = np.ma.masked_where(~np.isfinite(depths[i]) | (depths[i] <= 0), depths[i])
        im = ax.imshow(valid, cmap='Spectral', norm=Normalize(0, depth_max))
        ax.set_title(LABELS[i]); ax.axis('off')
        ax = axes[1, i + 1]
        p = clouds[i]
        ax.scatter(p[:, 0], p[:, 2], s=.5, c=COLORS[i], rasterized=True)
        endpoints = d['endpoints_' + s]
        if endpoints.shape != (2, 3) or not np.isfinite(endpoints).all():
            raise ValueError('Each endpoint array must contain two finite 3D points')
        ax.plot(endpoints[:, 0], endpoints[:, 2], 'r.-', lw=1.4)
        ax.set_title(f'Distance: {np.linalg.norm(endpoints[1]-endpoints[0]):.2f} m')
        ax.set(xlim=bounds[0], ylim=bounds[1], xlabel='x (m)', ylabel='z (m)', aspect='equal')
    fig.colorbar(im, ax=list(axes[0, 1:]), label='Camera-z depth (m)', shrink=.7)
    save(fig, out, 'nuscenes_depth')

    clouds = [d['cloud_' + s] for s in suffixes]
    allp = np.concatenate(clouds)
    bounds = [(np.nanmin(allp[:, k]), np.nanmax(allp[:, k])) for k in [0, 1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(title + ' | Reconstruction — shared metric BEV', fontsize=17)
    for ax, p, label, color in zip(axes, clouds, LABELS, COLORS):
        ax.scatter(p[:, 0], p[:, 1], s=.15, c=color, rasterized=True)
        ax.set(title=label, xlabel='x (m)', ylabel='y (m)', xlim=bounds[0], ylim=bounds[1], aspect='equal')
        ax.grid(alpha=.25)
    save(fig, out, 'nuscenes_reconstruction')

    poses = [d['trajectory_' + s] for s in suffixes]
    if any(p.shape != poses[0].shape for p in poses) or len(poses[0]) <= segment:
        raise ValueError('Trajectories must be synchronized Nx3 arrays longer than segment')
    gt = poses[0]
    traveled = np.r_[0, np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))]
    indices = np.arange(0, len(gt), segment)
    gt_lengths = np.linalg.norm(np.diff(gt[indices], axis=0), axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle(title + ' | Ego pose — scale-preserving common frame', fontsize=17)
    for p, label, color in zip(poses, LABELS, COLORS):
        length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        axes[0].plot(p[:, 0], p[:, 1], color=color, label=f'{label}: {length:.1f} m')
    for p, label, color in zip(poses[1:], LABELS[1:], COLORS[1:]):
        lengths = np.linalg.norm(np.diff(p[indices], axis=0), axis=1)
        valid = gt_lengths > .5
        axes[1].plot(traveled[indices[1:]][valid], lengths[valid]/gt_lengths[valid], '.-', color=color, label=label)
        error = np.linalg.norm(p-gt, axis=1)
        axes[2].plot(traveled, error, color=color, label=f'{label}: ATE {np.sqrt(np.mean(error**2)):.2f} m')
    axes[0].set(xlabel='x (m)', ylabel='y (m)', aspect='equal', title='Trajectory / path length')
    axes[1].axhline(1, color=COLORS[0], ls='--')
    axes[1].set(xlabel='GT traveled distance (m)', ylabel='Translation / GT translation', title='Segment scale ratio')
    axes[2].set(xlabel='GT traveled distance (m)', ylabel='Position error (m)', title='Position error')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.25)
    save(fig, out, 'nuscenes_ego_pose')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/figures/nuscenes_scale'))
    parser.add_argument('--title', default='nuScenes')
    parser.add_argument('--depth-max', type=float, default=80)
    parser.add_argument('--segment-frames', type=int, default=10)
    args = parser.parse_args()
    if args.segment_frames < 1 or args.depth_max <= 0:
        parser.error('segment-frames and depth-max must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.bundle:
        render(args.bundle, args.output, args.title, args.depth_max, args.segment_frames)
    else:
        preview(args.output)
