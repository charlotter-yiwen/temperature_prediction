"""
convert_tech_report.py
Convert TECH_REPORT.md to PDF using matplotlib (handles Chinese).
"""
import os, matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use('Agg')

TP_DIR = r"c:/Users/jkong/Documents/power brain_new/yiwen version/temperature_prediction"
MD_PATH = os.path.join(TP_DIR, "model_v3", "results_plan_a_physics", "TECH_REPORT.md")
PDF_PATH = os.path.join(TP_DIR, "model_v3", "results_plan_a_physics", "TECH_REPORT.pdf")

# Try to find a font that supports Chinese
import subprocess
font_paths = [
    "C:/Windows/Fonts/msyh.ttc",   # Microsoft YaHei
    "C:/Windows/Fonts/simsun.ttc",  # SimSun
]
font_prop = None
for fp in font_paths:
    if os.path.exists(fp):
        try:
            font_prop = matplotlib.font_manager.FontProperties(fname=fp)
            plt.rcParams['font.family'] = matplotlib.font_manager.FontProperties(fname=fp).get_name()
            plt.rcParams['axes.unicode_minus'] = False
            break
        except:
            pass

if font_prop is None:
    # fallback to default
    plt.rcParams['axes.unicode_minus'] = False

# Read markdown
with open(MD_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Build styled paragraphs
def md_to_plain(text):
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)         # italic
    text = re.sub(r'`(.*?)`', r'\1', text)           # inline code
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # links
    return text

def is_header(line, level):
    return line.strip().startswith('#' * level) and len(line.strip()) > level

def parse_md(lines):
    sections = []
    current_section = {'title': '', 'level': 0, 'content': []}
    in_code = False
    code_block = []

    for raw_line in lines:
        line = raw_line.rstrip('\n')

        # Code fence
        if line.strip().startswith('```'):
            if not in_code:
                in_code = True
                code_block = []
            else:
                in_code = False
                current_section['content'].append(('code', '\n'.join(code_block)))
                code_block = []
            continue
        if in_code:
            code_block.append(line)
            continue

        # Headers
        if is_header(line, 1):
            if current_section['content']:
                sections.append(current_section)
            current_section = {'title': md_to_plain(line.lstrip('#').strip()), 'level': 1, 'content': []}
            continue
        if is_header(line, 2):
            if current_section['content']:
                sections.append(current_section)
            current_section = {'title': md_to_plain(line.lstrip('#').strip()), 'level': 2, 'content': []}
            continue
        if is_header(line, 3):
            current_section['content'].append(('h3', md_to_plain(line.lstrip('#').strip())))
            continue
        if is_header(line, 4):
            current_section['content'].append(('h4', md_to_plain(line.lstrip('#').strip())))
            continue

        # Table - collect rows
        if line.strip().startswith('|'):
            row = [c.strip() for c in line.strip().strip('|').split('|')]
            if all(c.replace('-', '').replace(':', '') == '' for c in row):
                continue  # separator
            current_section['content'].append(('tr', row))
            continue

        # List
        if line.strip().startswith('- ') or line.strip().startswith('* '):
            current_section['content'].append(('li', md_to_plain(line.strip()[2:])))
            continue

        # Horizontal rule
        if line.strip() == '---':
            current_section['content'].append(('hr', ''))
            continue

        # Empty line
        if line.strip() == '':
            current_section['content'].append(('空', ''))
            continue

        # Paragraph
        current_section['content'].append(('p', md_to_plain(line.strip())))

    if current_section['content']:
        sections.append(current_section)
    return sections

sections = parse_md(lines)

# Render PDF
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

fig_width = 8.5  # inches (A4-ish)
with PdfPages(PDF_PATH) as pdf:

    # Page 1: Title
    fig = plt.figure(figsize=(fig_width, 11))
    ax = fig.add_axes([0.05, 0.3, 0.9, 0.6])
    ax.axis('off')
    ax.text(0.5, 0.85, 'Plan A + Physics Loss', fontsize=24, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.65, 'Technical Report', fontsize=16, ha='center', va='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.45, 'Model: plan_a_physics_phase2.pth  |  Params: 42.8M  |  Date: 2026-03-30',
            fontsize=10, ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.text(0.5, 0.25, 'Generalization Results', fontsize=13, ha='center', va='center',
            transform=ax.transAxes)
    ax.text(0.5, 0.12,
            '6-Component R2=0.9228  |  7-Component R2=0.9101  |  8-Component R2=0.8981  |  9-Component R2=0.4449',
            fontsize=11, ha='center', va='center', transform=ax.transAxes, fontweight='bold', color='darkblue')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Content pages
    for sec in sections:
        if sec['level'] == 1:
            fig = plt.figure(figsize=(fig_width, 11))
            ax = fig.add_axes([0.05, 0.85, 0.9, 0.1])
            ax.axis('off')
            ax.text(0.0, 0.5, sec['title'], fontsize=16, fontweight='bold', va='center')
            fig.add_axes(ax)
            ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.78])
            ax2.axis('off')
            y = 1.0
            for typ, content in sec['content']:
                if typ == '空':
                    y -= 0.015
                    continue
                if typ == 'h3':
                    y -= 0.04
                    ax2.text(0, y, content, fontsize=11, fontweight='bold')
                    y -= 0.03
                    continue
                if typ == 'h4':
                    y -= 0.03
                    ax2.text(0, y, content, fontsize=10, fontweight='bold')
                    y -= 0.025
                    continue
                if typ == 'li':
                    if y < 0.08:
                        break
                    ax2.text(0.02, y, '- ' + content[:120], fontsize=9)
                    y -= 0.025
                    continue
                if typ == 'p':
                    if y < 0.08:
                        break
                    ax2.text(0, y, content[:150], fontsize=9)
                    y -= 0.022
                    continue
                if typ == 'tr':
                    # skip in simple render
                    continue
                if typ == 'hr':
                    ax2.axhline(y=y, color='gray', linewidth=0.5)
                    y -= 0.02
                    continue
                if typ == 'code':
                    if y < 0.1:
                        break
                    ax2.text(0.01, y, content[:200], fontsize=6.5,
                             family='monospace', va='top',
                             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
                    y -= 0.08
                    continue
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        elif sec['level'] == 2:
            fig = plt.figure(figsize=(fig_width, 11))
            ax = fig.add_axes([0.05, 0.88, 0.9, 0.1])
            ax.axis('off')
            ax.text(0.0, 0.5, sec['title'], fontsize=13, fontweight='bold', va='center')
            fig.add_axes(ax)
            ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.80])
            ax2.axis('off')
            y = 0.99

            table_rows = []
            in_table = False

            def flush_table(ax2, table_rows, y):
                if not table_rows:
                    return y
                col_count = len(table_rows[0])
                row_h = 0.035
                col_w = 0.9 / col_count
                header = table_rows[0]
                for ri, row in enumerate(table_rows[:1]):  # header only
                    for ci, cell in enumerate(row):
                        cell = str(cell)[:25]
                        bg = '#404040' if ri == 0 else ('#f0f0f0' if ri % 2 == 0 else 'white')
                        fc = 'white' if ri == 0 else 'black'
                        rect = patches.Rectangle((ci * col_w, y - row_h), col_w, row_h,
                                                 linewidth=0.5, edgecolor='gray', facecolor=bg)
                        ax2.add_patch(rect)
                        ax2.text(ci * col_w + col_w / 2, y - row_h / 2, cell,
                                 fontsize=7, ha='center', va='center', color=fc, fontweight='bold' if ri == 0 else 'normal')
                    y -= row_h
                for ri, row in enumerate(table_rows[1:], 1):
                    for ci, cell in enumerate(row):
                        cell = str(cell)[:25]
                        bg = '#f0f0f0' if ri % 2 == 0 else 'white'
                        rect = patches.Rectangle((ci * col_w, y - row_h), col_w, row_h,
                                                 linewidth=0.3, edgecolor='lightgray', facecolor=bg)
                        ax2.add_patch(rect)
                        ax2.text(ci * col_w + col_w / 2, y - row_h / 2, cell,
                                 fontsize=6.5, ha='center', va='center')
                    y -= row_h
                y -= 0.02
                return y

            for typ, content in sec['content']:
                if typ == '空':
                    continue
                if typ == 'tr':
                    if not in_table:
                        in_table = True
                        table_rows = []
                    table_rows.append(content)
                    continue
                else:
                    if in_table:
                        y = flush_table(ax2, table_rows, y)
                        in_table = False
                        table_rows = []

                if typ == 'h3':
                    y -= 0.03
                    if y < 0.1:
                        break
                    ax2.text(0, y, content, fontsize=10, fontweight='bold')
                    y -= 0.025
                    continue
                if typ == 'h4':
                    y -= 0.02
                    if y < 0.1:
                        break
                    ax2.text(0, y, content, fontsize=9.5, fontweight='bold')
                    y -= 0.02
                    continue
                if typ == 'li':
                    if y < 0.1:
                        break
                    ax2.text(0.02, y, '- ' + content[:130], fontsize=8.5)
                    y -= 0.022
                    continue
                if typ == 'p':
                    if y < 0.1:
                        break
                    ax2.text(0, y, content[:150], fontsize=8.5)
                    y -= 0.02
                    continue
                if typ == 'hr':
                    ax2.axhline(y=y, color='gray', linewidth=0.5)
                    y -= 0.015
                    continue
                if typ == 'code':
                    if y < 0.15:
                        break
                    ax2.text(0.01, y, content[:250], fontsize=6.5,
                             family='monospace', va='top',
                             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
                    y -= 0.12
                    continue

            if in_table:
                flush_table(ax2, table_rows, y)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

print(f"[Saved] {PDF_PATH}")
