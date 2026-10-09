from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Initialize Widescreen Presentation (16:9)
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Styling Palette
C_BG = RGBColor(248, 249, 250)        # Off-white background
C_PRIMARY = RGBColor(26, 54, 93)      # Deep Navy Blue
C_SECONDARY = RGBColor(43, 108, 176)  # Medical Blue
C_TEXT = RGBColor(45, 55, 72)         # Charcoal Text
C_CARD_BG = RGBColor(255, 255, 255)   # Card BG
C_BORDER = RGBColor(226, 232, 240)    # Border Line

def set_slide_background(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = C_BG

def add_header(slide, title_text, category_text="SMART HOSPITAL RECOMMENDER SYSTEM"):
    set_slide_background(slide)
    
    # Subheader / Category
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = C_SECONDARY
    
    # Main Action Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = C_PRIMARY

slide_layout = prs.slide_layouts[6] # Blank Layout

# --- SLIDE 1: Title Slide ---
s1 = prs.slides.add_slide(slide_layout)
set_slide_background(s1)
tbox = s1.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(3.5))
tf = tbox.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Putting People Back at the Heart of Healthcare"
p.font.size = Pt(36)
p.font.bold = True
p.font.color.rgb = C_PRIMARY

p2 = tf.add_paragraph()
p2.text = "Designing a Smart Hospital Recommender System That Cares for Patients & Staff"
p2.font.size = Pt(20)
p2.font.color.rgb = C_SECONDARY
p2.space_before = Pt(15)

p3 = tf.add_paragraph()
p3.text = "Registration No: 23BML0066 | Case Study Defense"
p3.font.size = Pt(14)
p3.font.color.rgb = C_TEXT
p3.space_before = Pt(30)


# --- SLIDE 2: The Human Cost ---
s2 = prs.slides.add_slide(slide_layout)
add_header(s2, "When a Hospital is Fragmented, Everyone Suffers")

stories = [
    ("The Multi-Illness Patient", "Sent to an isolated single-specialty ward where staff aren't trained to manage secondary conditions, leading to care delays."),
    ("The Surgical Patient", "Prepped early at 6:00 AM, only to have elective surgery canceled last-minute because downstream beds saturated overnight."),
    ("The Overworked Nurse", "Forced into consecutive night shifts by rigid, manual spreadsheets—leading to chronic fatigue, error risks, and burnout.")
]

card_w, card_h, gap = Inches(3.64), Inches(4.8), Inches(0.4)
for i, (title, desc) in enumerate(stories):
    left = Inches(0.8) + i * (card_w + gap)
    card = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(1.6), card_w, card_h)
    card.fill.solid(); card.fill.fore_color.rgb = C_CARD_BG; card.line.color.rgb = C_BORDER
    
    tf_c = card.text_frame
    tf_c.word_wrap = True; tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = Inches(0.3)
    
    p = tf_c.paragraphs[0]
    p.text = title; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = C_PRIMARY
    
    p_d = tf_c.add_paragraph()
    p_d.text = desc; p_d.font.size = Pt(13); p_d.font.color.rgb = C_TEXT; p_d.space_before = Pt(14)


# --- SLIDE 3: System Philosophy ---
s3 = prs.slides.add_slide(slide_layout)
add_header(s3, "Moving from Rigid Rules to Empathetic Decision Support")

goals = [
    ("1. Patient-Centered Placement", "Assign patients to consolidated multi-specialty wards so nursing teams understand all their health needs."),
    ("2. Guaranteed Bed Availability", "Synchronize surgical schedules directly with inpatient bed capacity to eliminate last-minute hallway delays."),
    ("3. Fair Workforce Scheduling", "Forecast admission surges 14 days ahead to generate conflict-free, predictable shift rosters for staff.")
]

for i, (title, desc) in enumerate(goals):
    top = Inches(1.8) + i * Inches(1.6)
    card = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), top, Inches(11.73), Inches(1.3))
    card.fill.solid(); card.fill.fore_color.rgb = C_CARD_BG; card.line.color.rgb = C_BORDER
    
    tf_c = card.text_frame
    tf_c.word_wrap = True; tf_c.margin_left = Inches(0.3); tf_c.margin_top = Inches(0.2)
    
    p = tf_c.paragraphs[0]
    p.text = title; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_PRIMARY
    
    p_d = tf_c.add_paragraph()
    p_d.text = desc; p_d.font.size = Pt(13); p_d.font.color.rgb = C_TEXT; p_d.space_before = Pt(6)


# --- SLIDE 4: 3 Core Questions ---
s4 = prs.slides.add_slide(slide_layout)
add_header(s4, "Three Structural Steps to Human-Centered Care")

pillars = [
    ("Step 1: Ward Structure", "Where should patients go?", "Group departments into multi-specialty units based on patient illness co-occurrence."),
    ("Step 2: Capacity Planning", "How do we manage capacity?", "Link operating rooms to inpatient beds to guarantee downstream recovery space."),
    ("Step 3: Fair Staffing", "Who works when?", "Forecast patient demand in advance to create balanced, fair nurse shift rosters.")
]

for i, (step, question, body) in enumerate(pillars):
    left = Inches(0.8) + i * (card_w + gap)
    card = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(1.6), card_w, card_h)
    card.fill.solid(); card.fill.fore_color.rgb = C_CARD_BG; card.line.color.rgb = C_BORDER
    
    tf_c = card.text_frame
    tf_c.word_wrap = True; tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = Inches(0.3)
    
    p = tf_c.paragraphs[0]
    p.text = step.upper(); p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = C_SECONDARY
    
    p_q = tf_c.add_paragraph()
    p_q.text = f'"{question}"'; p_q.font.size = Pt(16); p_q.font.bold = True; p_q.font.color.rgb = C_PRIMARY; p_q.space_before = Pt(8)
    
    p_b = tf_c.add_paragraph()
    p_b.text = body; p_b.font.size = Pt(13); p_b.font.color.rgb = C_TEXT; p_b.space_before = Pt(12)


# --- SLIDE 5: Step 1 Ward Design ---
s5 = prs.slides.add_slide(slide_layout)
add_header(s5, "Step 1: Designing Wards Around Real Patient Needs")

box1 = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.5))
box1.fill.solid(); box1.fill.fore_color.rgb = C_CARD_BG; box1.line.color.rgb = C_BORDER
tf1 = box1.text_frame; tf1.word_wrap = True; tf1.margin_left = tf1.margin_top = Inches(0.3)
tf1.paragraphs[0].text = "The Human Problem"; tf1.paragraphs[0].font.size = Pt(18); tf1.paragraphs[0].font.bold = True; tf1.paragraphs[0].font.color.rgb = C_PRIMARY
p_b1 = tf1.add_paragraph()
p_b1.text = "Elderly patients often suffer from multiple conditions (e.g., Heart Disease + Diabetes). Traditional single-specialty wards separate care, causing high transfer penalties and specialty mismatches."; p_b1.font.size = Pt(13); p_b1.space_before = Pt(12)

box2 = s5.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.73), Inches(4.5))
box2.fill.solid(); box2.fill.fore_color.rgb = C_CARD_BG; box2.line.color.rgb = C_BORDER
tf2 = box2.text_frame; tf2.word_wrap = True; tf2.margin_left = tf2.margin_top = Inches(0.3)
tf2.paragraphs[0].text = "The Smart Solution & Impact"; tf2.paragraphs[0].font.size = Pt(18); tf2.paragraphs[0].font.bold = True; tf2.paragraphs[0].font.color.rgb = C_PRIMARY
p_b2 = tf2.add_paragraph()
p_b2.text = "The system uses diagnostic log mining to discover overlapping disease trends and recommends merged multi-specialty wards.\n\nResult: 63.8% reduction in care-mismatch penalties because nursing staff possess secondary specialty training."; p_b2.font.size = Pt(13); p_b2.space_before = Pt(12)


# --- SLIDE 6: Step 2 Capacity Planning ---
s6 = prs.slides.add_slide(slide_layout)
add_header(s6, "Step 2: Protecting Surgical Patients from Gridlock")

b1 = s6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.5))
b1.fill.solid(); b1.fill.fore_color.rgb = C_CARD_BG; b1.line.color.rgb = C_BORDER
tf1 = b1.text_frame; tf1.word_wrap = True; tf1.margin_left = tf1.margin_top = Inches(0.3)
tf1.paragraphs[0].text = "Surgical Bottlenecks"; tf1.paragraphs[0].font.size = Pt(18); tf1.paragraphs[0].font.bold = True; tf1.paragraphs[0].font.color.rgb = C_PRIMARY
p_b1 = tf1.add_paragraph()
p_b1.text = "When operating rooms and inpatient wards are scheduled independently, unexpected emergency arrivals saturate beds, forcing elective surgeries to be canceled at the last minute."; p_b1.font.size = Pt(13); p_b1.space_before = Pt(12)

b2 = s6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.73), Inches(4.5))
b2.fill.solid(); b2.fill.fore_color.rgb = C_CARD_BG; b2.line.color.rgb = C_BORDER
tf2 = b2.text_frame; tf2.word_wrap = True; tf2.margin_left = tf2.margin_top = Inches(0.3)
tf2.paragraphs[0].text = "Synchronized Air-Traffic Control"; tf2.paragraphs[0].font.size = Pt(18); tf2.paragraphs[0].font.bold = True; tf2.paragraphs[0].font.color.rgb = C_PRIMARY
p_b2 = tf2.add_paragraph()
p_b2.text = "Stochastic capacity models act as an air-traffic controller—approving elective surgeries only when downstream bed recovery is guaranteed, while maintaining daily buffers for emergency walk-ins.\n\nResult: Zero last-minute cancellations due to bed shortages."; p_b2.font.size = Pt(13); p_b2.space_before = Pt(12)


# --- SLIDE 7: Step 3 Staffing ---
s7 = prs.slides.add_slide(slide_layout)
add_header(s7, "Step 3: Respecting the Nursing Workforce")

s_metrics = [
    ("80% Less Admin Time", "Automates roster creation, letting nurse managers spend time with patients instead of spreadsheets."),
    ("23% Lower Overtime", "LSTM neural networks predict patient inflows 14 days in advance to match shift supply with demand."),
    ("41% Fewer Conflicts", "Optimization engines incorporate labor law rules, skill mix, shift preferences, and fairness math.")
]

for i, (title, desc) in enumerate(s_metrics):
    left = Inches(0.8) + i * (card_w + gap)
    card = s7.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(1.6), card_w, card_h)
    card.fill.solid(); card.fill.fore_color.rgb = C_CARD_BG; card.line.color.rgb = C_BORDER
    
    tf_c = card.text_frame; tf_c.word_wrap = True; tf_c.margin_left = tf_c.margin_top = Inches(0.3)
    p = tf_c.paragraphs[0]
    p.text = title; p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = C_PRIMARY
    
    p_d = tf_c.add_paragraph()
    p_d.text = desc; p_d.font.size = Pt(13); p_d.font.color.rgb = C_TEXT; p_d.space_before = Pt(14)


# --- SLIDE 8: Data Privacy & Local AI ---
s8 = prs.slides.add_slide(slide_layout)
add_header(s8, "High Intelligence with Zero Privacy Risk")

privacy_features = [
    ("100% Local Processing", "Deploy local open-source models (e.g., Llama-3) on hospital servers. Patient data never touches commercial cloud APIs."),
    ("Plain-English Interaction", "Administrators interact using natural language prompts without needing complex database coding skills."),
    ("Bulletproof Reliability", "Automated Round-Robin fallback logic guarantees 100% system availability if primary optimization servers timeout.")
]

for i, (title, desc) in enumerate(privacy_features):
    top = Inches(1.8) + i * Inches(1.6)
    card = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), top, Inches(11.73), Inches(1.3))
    card.fill.solid(); card.fill.fore_color.rgb = C_CARD_BG; card.line.color.rgb = C_BORDER
    
    tf_c = card.text_frame; tf_c.word_wrap = True; tf_c.margin_left = Inches(0.3); tf_c.margin_top = Inches(0.2)
    p = tf_c.paragraphs[0]; p.text = title; p.font.size = Pt(16); p.font.bold = True; p.font.color.rgb = C_PRIMARY
    p_d = tf_c.add_paragraph(); p_d.text = desc; p_d.font.size = Pt(13); p_d.font.color.rgb = C_TEXT; p_d.space_before = Pt(6)


# --- SLIDE 9: Results Table ---
s9 = prs.slides.add_slide(slide_layout)
add_header(s9, "Measurable Human & Operational Results")

table_shape = s9.shapes.add_table(4, 4, Inches(0.8), Inches(1.8), Inches(11.73), Inches(4.5))
table = table_shape.table
table.columns[0].width = Inches(2.2); table.columns[1].width = Inches(3.0)
table.columns[2].width = Inches(3.8); table.columns[3].width = Inches(2.73)

headers = ["Operational Level", "Traditional Approach", "Smart Recommender Solution", "Human & Metric Impact"]
data = [
    ["Macro: Ward Setup", "Isolated single-specialty wards", "Co-morbidity ward merging via diagnostic mining", "63.8% drop in care-mismatch errors"],
    ["Operational: Capacity", "ORs and beds planned independently", "Stochastic capacity gatekeeping (SAA/DRO)", "Zero last-minute surgical cancellations"],
    ["Micro: Workforce Roster", "Manual, unfair spreadsheets", "LSTM demand forecasting + Genetic Rostering", "80% less admin work; 23% lower overtime"]
]

for col_idx, header in enumerate(headers):
    cell = table.cell(0, col_idx); cell.text = header; cell.fill.solid(); cell.fill.fore_color.rgb = C_PRIMARY
    for p in cell.text_frame.paragraphs:
        p.font.bold = True; p.font.color.rgb = RGBColor(255, 255, 255); p.font.size = Pt(13)

for row_idx, row_data in enumerate(data):
    for col_idx, cell_value in enumerate(row_data):
        cell = table.cell(row_idx + 1, col_idx); cell.text = cell_value; cell.fill.solid()
        cell.fill.fore_color.rgb = C_CARD_BG if row_idx % 2 == 0 else C_BG
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12); p.font.color.rgb = C_TEXT
            if col_idx == 3: p.font.bold = True; p.font.color.rgb = C_SECONDARY


# --- SLIDE 10: Conclusion & Defense Takeaway ---
s10 = prs.slides.add_slide(slide_layout)
add_header(s10, "Conclusion: Technology in Service of Care")

box_c = s10.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.73), Inches(4.5))
box_c.fill.solid(); box_c.fill.fore_color.rgb = C_CARD_BG; box_c.line.color.rgb = C_BORDER
tf_c = box_c.text_frame; tf_c.word_wrap = True; tf_c.margin_left = Inches(0.4); tf_c.margin_top = Inches(0.4)

p = tf_c.paragraphs[0]
p.text = "A Smarter Structure Means Better Patient Care"; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = C_PRIMARY

p2 = tf_c.add_paragraph()
p2.text = "• Alignment over Isolation: Smart healthcare isn't about buying expensive hardware—it is about connecting fragmented operational decisions.\n\n• Unified 3-Stage Pipeline: By linking physical ward layout, bed capacity, and nurse shift rosters into a single supportive system, we transform chaotic hospital operations into a predictable, human-centered environment.\n\n• Ready for Defense: The proposed framework directly resolves the core trade-offs across all 10 academic benchmark papers."; p2.font.size = Pt(14); p2.font.color.rgb = C_TEXT; p2.space_before = Pt(16)

# Save
output_path = "Smart_Hospital_Recommender_Full_10_Slides.pptx"
prs.save(output_path)
print(f"Full 10-Slide Presentation saved as '{output_path}'.")