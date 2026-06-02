"""
SUMMA Solar Radiation Processing — slide diagram
Three panels, one per processing stage.
Run: python summa_solar_radiation_slide.py
Output: summa_solar_radiation_slide.png (same directory)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Arc, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── palette ──────────────────────────────────────────────────────────────────
SUN    = '#FFB300'
SKY    = '#B3E5FC'
GRASS  = '#388E3C'
SNOW   = '#E1F5FE'
SOIL   = '#795548'
DIRECT = '#E65100'
DIFUSE = '#1565C0'
CANOPY = '#2E7D32'
PURPLE = '#6A1B9A'
GREY   = '#546E7A'

def arrow(ax, xy, xytext, color, lw=2.0, style='->'):
    ax.annotate('', xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=5)

def box(ax, xy, w, h, fc, ec, text, fontsize=8.5, tc='black', bold=False,
        pad=0.15, alpha=1.0):
    r = FancyBboxPatch(xy, w, h, boxstyle=f"round,pad={pad}",
                       facecolor=fc, edgecolor=ec, linewidth=2, alpha=alpha, zorder=3)
    ax.add_patch(r)
    fw = 'bold' if bold else 'normal'
    ax.text(xy[0]+w/2, xy[1]+h/2, text,
            ha='center', va='center', fontsize=fontsize,
            color=tc, fontweight=fw, zorder=4)

# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 7.5))
gs  = GridSpec(1, 3, figure=fig, wspace=0.38,
               left=0.03, right=0.98, top=0.88, bottom=0.06)

fig.suptitle("SUMMA Solar Radiation Processing Pipeline",
             fontsize=15, fontweight='bold', y=0.96)

# ════════════════════════════════════════════════════════════════════════════
# PANEL 1 — Terrain Geometry  (sunGeomtry.f90 : CLRSKY_RAD)
# ════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
ax1.set_aspect('equal'); ax1.axis('off')
ax1.set_title("① Terrain Geometry\nsunGeomtry.f90 · CLRSKY_RAD",
              fontsize=10.5, fontweight='bold', pad=6)

# sky fill
ax1.fill_between([0, 10], [4.5, 4.5], [10, 10], color=SKY, alpha=0.25, zorder=0)

# tilted HRU surface
sx = np.array([0.5, 9.5])
sy = np.array([3.5, 5.5])
ax1.fill_between(sx, sy, 0, color=GRASS, alpha=0.55, zorder=1)
ax1.plot(sx, sy, color='#1B5E20', lw=2.5, zorder=2)

# horizontal reference (dashed)
ax1.plot([1.0, 9.0], [4.2, 4.2], 'k--', lw=1.2, alpha=0.4, zorder=2)
ax1.text(9.2, 4.2, 'horizontal', fontsize=7, va='center', color=GREY)

# sun
sun_patch = plt.Circle((7.8, 8.6), 0.55, color=SUN, zorder=6)
ax1.add_patch(sun_patch)
for ang in np.linspace(0, 2*np.pi, 8, endpoint=False):
    ax1.plot([7.8 + 0.55*np.cos(ang), 7.8 + 1.0*np.cos(ang)],
             [8.6 + 0.55*np.sin(ang), 8.6 + 1.0*np.sin(ang)],
             color=SUN, lw=1.8, zorder=5)

# hit point on slope
hx, hy = 5.0, 4.35

# solar beam (direct)
arrow(ax1, (hx, hy), (7.35, 8.15), DIRECT, lw=2.8)
ax1.text(6.9, 6.6, 'direct\nbeam', fontsize=8, color=DIRECT, ha='center')

# zenith (vertical)
ax1.plot([hx, hx], [hy, 7.8], ':', color=GREY, lw=1.5, zorder=2)
ax1.text(hx+0.12, 6.2, 'zenith', fontsize=7.5, color=GREY, rotation=90, va='center')

# surface normal
beta_rad = np.arctan(2.0/9.0)          # slope angle of our drawn surface
nx = -np.sin(beta_rad); ny = np.cos(beta_rad)
scale = 2.2
ax1.annotate('', xy=(hx + scale*nx, hy + scale*ny), xytext=(hx, hy),
             arrowprops=dict(arrowstyle='->', color=CANOPY, lw=1.8,
                             linestyle='dashed'), zorder=4)
ax1.text(hx + scale*nx - 0.9, hy + scale*ny + 0.1,
         'surface\nnormal', fontsize=7.5, color=CANOPY)

# zenith-angle arc
za = Arc((hx, hy), 2.8, 2.8, angle=0, theta1=64, theta2=90,
         color=GREY, lw=1.5)
ax1.add_patch(za)
ax1.text(hx+0.55, hy+1.55, 'θ_z', fontsize=9, color=GREY)

# incidence-angle arc
ia = Arc((hx, hy), 2.0, 2.0, angle=0, theta1=40, theta2=64,
         color=DIRECT, lw=1.5)
ax1.add_patch(ia)
ax1.text(hx-0.35, hy+1.1, 'θ_i', fontsize=9, color=DIRECT)

# slope label
ax1.text(2.0, 1.8, 'HRU slope β\nHRU aspect α',
         fontsize=8, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor=GRASS, alpha=0.9))

# sunrise arc
sa = Arc((5, 4.35), 8.5, 8.5, angle=0, theta1=8, theta2=172,
         color=SUN, lw=1.2, linestyle='--', alpha=0.5)
ax1.add_patch(sa)
ax1.text(5, 8.7, 'integrated over Δt (hours → radians)',
         fontsize=7.5, ha='center', color='#B8860B', style='italic')

# HRI equation box
box(ax1, (0.8, 0.25), 8.4, 1.1,
    fc='#FFFDE7', ec=SUN,
    text='HRI = ∫cos(θᵢ)dt / (cos(β)·Δt)    →    cosZenith',
    fontsize=8.5, bold=False)


# ════════════════════════════════════════════════════════════════════════════
# PANEL 2 — Spectral Partitioning  (derivforce.f90)
# ════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title("② Spectral Partitioning\nderivforce.f90",
              fontsize=10.5, fontweight='bold', pad=6)

# ERA5 input
box(ax2, (2.5, 8.3), 5.0, 1.1, fc='#FFF9C4', ec=SUN,
    text='SWRadAtm  (ERA5, W m⁻²)', fontsize=9, bold=True)

arrow(ax2, (5, 8.3), (5, 7.65), GREY, lw=2)
ax2.text(5.15, 7.93, 'cosZenith\n(from Stage 1)',
         fontsize=7.5, color=GREY, va='center')

# NL parameterisation
box(ax2, (1.5, 6.6), 7.0, 0.95, fc='#E8EAF6', ec='#3949AB',
    text='Nijssen-Lettenmaier (1999):   f_direct = f₀·cosθ / (cosθ + k)',
    fontsize=8.0)

# split to direct / diffuse
arrow(ax2, (2.5, 5.85), (3.2, 6.6), DIRECT, lw=2)
arrow(ax2, (7.5, 5.85), (6.8, 6.6), DIFUSE, lw=2)

box(ax2, (0.3, 4.55), 4.0, 1.2, fc='#FFF3E0', ec=DIRECT,
    text='Direct\n(cosθ → 1 at noon)', fontsize=8.5,
    tc=DIRECT, bold=True)
box(ax2, (5.7, 4.55), 4.0, 1.2, fc='#E3F2FD', ec=DIFUSE,
    text='Diffuse\n(cosθ → 0 near dawn/dusk)', fontsize=8.5,
    tc=DIFUSE, bold=True)

# VIS / NIR split below each
for cx, col_vis, col_nir, label_vis, label_nir in [
        (2.3, '#FF8F00', '#BF360C', 'Direct\nVIS', 'Direct\nNIR'),
        (7.7, '#1565C0', '#0D47A1', 'Diffuse\nVIS', 'Diffuse\nNIR')]:
    arrow(ax2, (cx - 0.9, 3.25), (cx - 0.4, 4.55), col_vis, lw=1.8)
    arrow(ax2, (cx + 0.9, 3.25), (cx + 0.4, 4.55), col_nir, lw=1.8)
    ec_v = col_vis; ec_n = col_nir
    box(ax2, (cx - 1.9, 2.15), 1.8, 1.0, fc='white', ec=ec_v,
        text=label_vis, fontsize=8.0, tc=col_vis, bold=True, pad=0.1)
    box(ax2, (cx + 0.1, 2.15), 1.8, 1.0, fc='white', ec=ec_n,
        text=label_nir, fontsize=8.0, tc=col_nir, bold=True, pad=0.1)

ax2.text(5, 1.5, '4 spectral fluxes → canopy module (Stage 3)',
         ha='center', fontsize=8.5,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFDE7',
                   edgecolor=SUN, alpha=0.95))
ax2.text(5, 0.65, 'split further by Frad_vis (~50% vis / 50% NIR)',
         ha='center', fontsize=7.5, color=GREY, style='italic')


# ════════════════════════════════════════════════════════════════════════════
# PANEL 3 — Canopy & Ground Partitioning  (vegSWavRad.f90)
# ════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[2])
ax3.set_xlim(0, 10); ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title("③ Canopy & Ground Partitioning\nvegSWavRad.f90",
              fontsize=10.5, fontweight='bold', pad=6)

# sky
ax3.fill_between([0, 10], [8.9, 8.9], [10, 10], color=SKY, alpha=0.30, zorder=0)

# incoming SW
arrow(ax3, (4.5, 8.15), (4.5, 9.4), DIRECT, lw=3.5)
ax3.text(5.05, 8.85, 'SW_in (4 fluxes)', fontsize=8.5, va='center', color=DIRECT)

# reflected upward
arrow(ax3, (1.8, 9.3), (1.8, 8.15), GREY, lw=2)
ax3.text(0.9, 9.55, 'Reflected\nto space', ha='center', fontsize=7.5, color=GREY)

# canopy
box(ax3, (1.0, 6.9), 8.0, 1.1, fc=CANOPY, ec='#1B5E20',
    text='Canopy   (LAI, SAI, wet fraction)',
    fontsize=9, tc='white', bold=True, alpha=0.75)

# canopy absorbed side label
arrow(ax3, (9.55, 7.45), (9.0, 7.45), CANOPY, lw=2, style='-|>')
ax3.text(9.6, 7.45,
         'Q_canopy\nabsorbed',
         fontsize=8, color=CANOPY, fontweight='bold', va='center')

# method selection inside a small banner
box(ax3, (1.3, 6.4), 7.4, 0.42, fc='#F3E5F5', ec=PURPLE,
    text='canopySrad:  CLM_2stream | UEB_2stream | NL_scatter | BeersLaw | noah_mp',
    fontsize=7.0, tc=PURPLE, pad=0.05, bold=False)

# transmitted below canopy
arrow(ax3, (4.5, 5.4), (4.5, 6.4), DIRECT, lw=2.5, style='->')
ax3.text(5.0, 5.9, 'transmitted', fontsize=7.5, color=GREY, style='italic', va='center')

# snow surface
box(ax3, (1.0, 4.35), 8.0, 0.95, fc=SNOW, ec='#81D4FA',
    text='Snow / Ground Surface',
    fontsize=9, tc='#01579B', bold=True)

# ground albedo subroutine note
ax3.text(5, 4.1, 'gndAlbedo: snow-fraction weighted albedo (spectral)',
         ha='center', fontsize=7.5, color=GREY, style='italic')

# soil layer
box(ax3, (1.0, 3.05), 8.0, 0.95, fc=SOIL, ec='#4E342E',
    text='Soil', fontsize=9, tc='white', bold=True, alpha=0.8)

# ground absorbed
arrow(ax3, (9.55, 4.82), (9.0, 4.82), '#01579B', lw=2, style='-|>')
ax3.text(9.6, 4.82, 'Q_ground\nabsorbed',
         fontsize=8, color='#01579B', fontweight='bold', va='center')

# sunlit/shaded split box at bottom
box(ax3, (0.5, 1.55), 9.0, 1.35, fc='#FFFDE7', ec=SUN,
    text='cosZenith → sunlit/shaded LAI split → PAR for photosynthesis\n'
         'scalarCanopySunlitPAR  |  scalarCanopyShadedPAR',
    fontsize=8.0, bold=False)

ax3.text(5, 0.7, 'drives Ball-Berry / Jarvis stomatal resistance',
         ha='center', fontsize=7.5, color=GREY, style='italic')


# ── save ─────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / 'summa_solar_radiation_slide.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved → {out}")
plt.show()
