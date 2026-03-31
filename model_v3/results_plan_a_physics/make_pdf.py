"""
Markdown to PDF using matplotlib - reliable text rendering.
"""
import os, re, matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

TP_DIR = r"c:/Users/jkong/Documents/power brain_new/yiwen version/temperature_prediction"
MD_PATH = os.path.join(TP_DIR, "model_v3", "results_plan_a_physics", "TECH_REPORT_EN.md")
PDF_PATH = os.path.join(TP_DIR, "model_v3", "results_plan_a_physics", "TECH_REPORT.pdf")

with open(MD_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# Parse into styled blocks
blocks = []
in_code = False
code_block = []

for line in content.split('\n'):
    if line.startswith('```'):
        if not in_code:
            in_code = True
            code_block = []
        else:
            in_code = False
            blocks.append(('code', '\n'.join(code_block)))
            code_block = []
        continue
    if in_code:
        code_block.append(line)
        continue

    if line.startswith('# '):
        blocks.append(('h1', line[2:]))
    elif line.startswith('## '):
        blocks.append(('h2', line[3:]))
    elif line.startswith('### '):
        blocks.append(('h3', line[4:]))
    elif line.startswith('#### '):
        blocks.append(('h4', line[5:]))
    elif line.strip() == '---':
        blocks.append(('hr', ''))
    elif line.strip().startswith('|'):
        blocks.append(('tr', [c.strip() for c in line.strip().strip('|').split('|')]))
    elif line.strip().startswith('- ') or line.strip().startswith('* '):
        blocks.append(('li', line.strip()[2:]))
    elif line.strip() == '':
        blocks.append(('space', ''))
    else:
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        blocks.append(('p', text))

# Render
page_w = 8.5
margin = 0.5
text_w = page_w - 2 * margin
line_h = 0.045
fig_h = 11

def new_page():
    fig = plt.figure(figsize=(page_w, fig_h))
    ax = fig.add_axes([margin / page_w, 0.05, text_w / page_w, 0.90])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax

def check_space(ax, y, needed):
    if y - needed < 0.05:
        return False
    return True

with PdfPages(PDF_PATH) as pdf:
    fig, ax = new_page()
    y = 0.97

    i = 0
    while i < len(blocks):
        typ, content = blocks[i]

        if typ == 'h1':
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            fig, ax = new_page()
            y = 0.97
            ax.text(0.5, 0.5, content, fontsize=16, fontweight='bold',
                    ha='center', va='center', transform=ax.transAxes)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            fig, ax = new_page()
            y = 0.97
            i += 1
            continue

        elif typ == 'h2':
            if not check_space(ax, y, 0.10):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                fig, ax = new_page()
                y = 0.97
            y -= 0.02
            ax.add_patch(patches.Rectangle((0, y - 0.03), 1, 0.04,
                             linewidth=0, facecolor='#404040', transform=ax.transAxes))
            ax.text(0.02, y - 0.015, content, fontsize=11, fontweight='bold',
                    va='center', color='white', transform=ax.transAxes)
            y -= 0.05
            i += 1
            continue

        elif typ == 'h3':
            if not check_space(ax, y, 0.08):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                fig, ax = new_page()
                y = 0.97
            y -= 0.02
            ax.text(0, y, content, fontsize=10, fontweight='bold', transform=ax.transAxes)
            y -= 0.04
            i += 1
            continue

        elif typ == 'h4':
            if not check_space(ax, y, 0.06):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                fig, ax = new_page()
                y = 0.97
            y -= 0.01
            ax.text(0, y, content, fontsize=9, fontweight='bold', transform=ax.transAxes)
            y -= 0.03
            i += 1
            continue

        elif typ == 'hr':
            y -= 0.01
            ax.plot([0, 1], [y, y], color='gray', linewidth=0.5, transform=ax.transAxes)
            y -= 0.02
            i += 1
            continue

        elif typ == 'tr':
            # Collect table rows
            table_data = []
            while i < len(blocks) and blocks[i][0] == 'tr':
                table_data.append(blocks[i][1])
                i += 1
            if not table_data:
                continue

            # Check space
            rows_needed = len(table_data) * 0.04 + 0.02
            if not check_space(ax, y, rows_needed):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                fig, ax = new_page()
                y = 0.97

            y -= 0.01
            col_count = len(table_data[0])
            col_w = 1.0 / col_count
            row_h = 0.03

            for ri, row in enumerate(table_data):
                for ci, cell in enumerate(row):
                    rx, ry = ci * col_w, y - row_h
                    bg = '#404040' if ri == 0 else ('#f0f0f0' if ri % 2 == 0 else 'white')
                    fc = 'white' if ri == 0 else 'black'
                    ax.add_patch(patches.Rectangle((rx, ry), col_w, row_h,
                                     linewidth=0.3, edgecolor='gray', facecolor=bg,
                                     transform=ax.transAxes))
                    ax.text(rx + col_w / 2, ry + row_h / 2, cell[:25],
                           fontsize=6.5, ha='center', va='center', color=fc,
                           fontweight='bold' if ri == 0 else 'normal',
                           transform=ax.transAxes)
                y -= row_h
            y -= 0.02
            continue

        elif typ == 'li':
            lines_needed = 0.035
            if not check_space(ax, y, lines_needed):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                fig, ax = new_page()
                y = 0.97
            ax.text(0.02, y, '- ' + content[:100], fontsize=8.5,
                   va='top', transform=ax.transAxes)
            y -= 0.035
            i += 1
            continue

        elif typ == 'space':
            y -= 0.015
            i += 1
            continue

        elif typ == 'code':
            lines = content.split('\n')
            code_lines = len(lines)
            code_needed = code_lines * 0.025 + 0.02
            if not check_space(ax, y, code_needed):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                fig, ax = new_page()
                y = 0.97
            y -= 0.01
            for line in lines[:30]:
                ax.text(0.01, y, line[:120], fontsize=6, family='monospace',
                       va='top', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9, linewidth=0))
                y -= 0.025
            y -= 0.01
            i += 1
            continue

        elif typ == 'p':
            if not check_space(ax, y, 0.05):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                fig, ax = new_page()
                y = 0.97
            ax.text(0, y, content[:200], fontsize=8.5, va='top', transform=ax.transAxes)
            y -= 0.032
            i += 1
            continue

        else:
            i += 1

    # Save last page
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

print(f"[Saved] {PDF_PATH}")
print(f"Size: {os.path.getsize(PDF_PATH) / 1024:.1f} KB")
