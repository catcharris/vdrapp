import matplotlib.pyplot as plt
import io
import numpy as np

def draw_staff_and_note(note_name, clef_type="Treble"):
    """
    Draws a simple musical staff with a single note.
    note_name: "F4", "E4", "D4"
    clef_type: "Treble" or "Bass"
    """
    # Fix Dark Mode: Use White background (Paper style)
    fig, ax = plt.subplots(figsize=(3, 1.8), facecolor='white')
    ax.set_axis_off()
    
    # Draw Staff Lines
    y_lines = [0, 2, 4, 6, 8]
    for y in y_lines:
        ax.hlines(y, xmin=0, xmax=10, colors='black', linewidth=1)

    # --- VECTOR GRAPHICS FOR CLEFS ---
    from matplotlib.path import Path
    import matplotlib.patches as patches

    def draw_svg_path(ax, path_data, scale=1, offset=(0,0), color='black'):
        vertices = []
        codes = []
        parts = path_data.split()
        i = 0
        while i < len(parts):
            cmd = parts[i]
            if cmd == 'M':
                codes.append(Path.MOVETO)
                vertices.append((float(parts[i+1]), float(parts[i+2])))
                i += 3
            elif cmd == 'L':
                codes.append(Path.LINETO)
                vertices.append((float(parts[i+1]), float(parts[i+2])))
                i += 3
            elif cmd == 'C':
                codes.append(Path.CURVE4)
                vertices.append((float(parts[i+1]), float(parts[i+2])))
                codes.append(Path.CURVE4)
                vertices.append((float(parts[i+3]), float(parts[i+4])))
                codes.append(Path.CURVE4)
                vertices.append((float(parts[i+5]), float(parts[i+6])))
                i += 7
            elif cmd == 'Z':
                codes.append(Path.CLOSEPOLY)
                vertices.append((0,0)) # Ignored
                i += 1
            else:
                # Handle implicit commands if needed, but for now stick to simple SVG subset
                i += 1
        
        if not vertices: return

        # Create Path
        path = Path(vertices, codes)
        # Scale and Translate
        # Note: SVG y is usually down. Matplotlib y is up. Flip y?
        # Let's assume input paths are flipped or handle flip.
        
        patch = patches.PathPatch(path, facecolor=color, lw=0)
        # Use simple transform via Affine2D? 
        # Easier: Manually transform vertices before creating Path?
        # Or Just use ax.add_patch with transform...
        
        # Let's do simple vertex transformation
        t_verts = []
        for v in vertices:
            # Flip Y (SVG coords usually top-left)
            vx = v[0] * scale + offset[0]
            vy = -v[1] * scale + offset[1] 
            t_verts.append((vx, vy))
            
        final_path = Path(t_verts, codes)
        ax.add_patch(patches.PathPatch(final_path, facecolor=color, lw=0))

    # Simplified SVG Paths (Approximate for visual quality)
    # Treble Clef Path
    treble_path = "M 152 645 C 139 635 125 617 120 603 C 113 584 116 565 130 541 C 141 522 159 500 181 479 C 196 464 200 457 197 453 C 191 445 161 453 145 466 C 104 502 81 556 81 615 C 81 672 110 721 157 743 C 187 758 238 757 265 741 C 302 720 323 681 323 635 C 323 584 290 531 234 492 C 220 482 220 482 219 403 L 218 333 L 209 344 C 183 376 139 397 99 400 C 47 404 9 369 9 316 C 9 270 34 230 75 210 C 97 200 137 197 160 204 C 172 208 184 213 186 216 C 189 219 190 60 189 -138 L 188 -163 L 174 -155 C 166 -150 152 -136 142 -123 C 122 -97 119 -84 122 -59 C 124 -40 132 -26 148 -14 C 167 1 199 4 220 -6 C 243 -17 254 -46 248 -79 C 246 -94 235 -118 226 -128 L 210 -144 L 211 -116 C 213 -73 194 -46 163 -46 C 147 -46 140 -53 140 -69 C 140 -74 145 -87 151 -97 C 168 -124 204 -148 227 -148 C 236 -148 244 -145 244 -141 C 244 -137 241 -122 237 -107 C 227 -63 248 -24 286 -17 C 326 -10 361 -35 373 -80 C 379 -102 376 -133 366 -153 C 347 -192 312 -214 271 -214 C 255 -214 237 -212 231 -209 C 218 -202 218 -202 218 206 L 218 614 C 218 672 216 673 184 661 C 166 654 159 650 152 645 Z M 190 248 C 186 242 165 233 143 227 C 114 220 84 227 65 245 C 47 263 45 296 61 324 C 76 350 108 365 144 364 C 180 363 216 339 219 301 C 220 286 219 285 190 248 Z"
    
    # Bass Clef Path (Simplified)
    bass_path = "M 120 540 C 70 520 40 470 40 410 C 40 330 95 270 160 270 C 210 270 240 310 240 360 C 240 430 180 540 110 540 Z M 270 470 C 270 485 285 500 300 500 C 315 500 330 485 330 470 C 330 455 315 440 300 440 C 285 440 270 455 270 470 Z M 270 390 C 270 405 285 420 300 420 C 315 420 330 405 330 390 C 330 375 315 360 300 360 C 285 360 270 375 270 390 Z"

    # Draw Clef (Using SVG Paths)
    if clef_type == "Treble":
        # Scale and Position for Treble Clef
        # Original SVG path is large coordinates (?~700). Staff lines are 0..8.
        # Needs roughly 0.01 scale.
        draw_svg_path(ax, treble_path, scale=0.012, offset=(1, 2.5)) 
        
        # Note Mapping (Treble)
        # E4 = Line 1 (y=0)
        # F4 = Space 1 (y=1)
        base_y = 0 
        base_note_idx = 0 
        
        note_y_map = {
            "C4": -4, "D4": -2, "E4": 0, "F4": 1, "G4": 2, "A4": 3, "B4": 4, "C5": 5, "D5": 6, "E5": 7, "F5": 8
        }
        
    elif clef_type == "Bass":
        # Scale and Position for Bass Clef
        # F3 is Line 4 (y=6). Dots flank y=6.
        draw_svg_path(ax, bass_path, scale=0.015, offset=(1, 5))
        
        # Note Mapping (Bass)
        # G2(Line 1)=0. A2=1. B2=2. C3=3. D3=4. E3=5. F3(Line 4)=6. G3=7. A3(Line 5)=8. B3=9. C4=10. D4=11. E4=12. F4=13.
        # Step=1.
        note_y_map = {
            "G2": 0, "A2": 1, "B2": 2, "C3": 3, "D3": 4, "E3": 5, "F3": 6, "G3": 7, "A3": 8, "B3": 9, "C4": 10, "D4": 11, "E4": 12, "F4": 13
        }
    
    # Note Position Lookup
    y_pos = note_y_map.get(note_name, 0)
    
    # Draw Note Head (Black)
    ax.scatter([5], [y_pos], s=400, color='black', marker='o')
    
    # Draw Ledger Lines
    # Staff Range: 0 to 8.
    # Below Staff: -2, -4 -> Draw lines
    # Above Staff: 10, 12, 14 -> Draw lines
    
    # Check bottom ledgers
    curr_y = -2
    while curr_y >= y_pos:
        ax.hlines(curr_y, 4, 6, colors='black', linewidth=1)
        curr_y -= 2
        
    # Check top ledgers
    curr_y = 10
    while curr_y <= y_pos:
        ax.hlines(curr_y, 4, 6, colors='black', linewidth=1)
        curr_y += 2
    
    # Label Note Name below
    ax.text(5, y_pos - 4.5, note_name, fontsize=12, ha='center', color='black', fontweight='bold')
    
    ax.set_ylim(-6, 16)
    ax.set_xlim(0, 8)
    
    # Save to buffer with White Background (NOT transparent)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf
