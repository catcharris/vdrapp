import matplotlib.pyplot as plt
import io

def draw_staff_and_note(note_name):
    """
    Draws a simple musical staff with a single note.
    note_name: "F4", "E4", "D4"
    """
    # Fix Dark Mode: Use White background (Paper style)
    fig, ax = plt.subplots(figsize=(3, 1.5), facecolor='white')
    ax.set_axis_off()
    
    # Draw Staff Lines (y=0, 2, 4, 6, 8)
    y_lines = [0, 2, 4, 6, 8]
    for y in y_lines:
        ax.hlines(y, xmin=0, xmax=10, colors='black', linewidth=1)
        
    # Draw Clef - Removed Unicode '𝄞' due to rendering issues
    # Just draw a vertical bar at the start to simulate a system line
    ax.vlines(0, 0, 8, colors='black', linewidth=2)
    
    # Note Position
    note_y_map = {
        "D4": -1, # Below E4
        "E4": 0,  # Line 1
        "F4": 1,  # Space 1
        "G4": 2,
        "A4": 3,
        "C4": -3 
    }
    
    y_pos = note_y_map.get(note_name, 0)
    
    # Draw Note Head (Black)
    ax.scatter([5], [y_pos], s=400, color='black', marker='o')
    
    # Draw Ledger Line if note is C4 (or D4 if we want to be strict, but D4 hangs)
    if note_name == "C4":
        ax.hlines(-2, 4, 6, colors='black', linewidth=1)
    
    # Label Note Name below
    ax.text(5, y_pos - 3.5, note_name, fontsize=12, ha='center', color='black', fontweight='bold')
    
    ax.set_ylim(-5, 12)
    ax.set_xlim(-1, 11)
    
    # Save to buffer with White Background (NOT transparent)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf
