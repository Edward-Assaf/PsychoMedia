import os
import textwrap
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Personality Archetypes mapping
ARCHETYPES = {
    'sOPN': {
        'title': 'The Visionary (High Openness)',
        'color': (6, 182, 212),  # Teal #06B6D4
        'iconName': 'openness.png',
        'description': (
            "You display high intellectual curiosity, imagination, and a natural preference for "
            "novelty and variety. Your communication reflects complex concepts, creative thoughts, "
            "and an appreciation for artistic and philosophical themes."
        )
    },
    'sCON': {
        'title': 'The Organizer (High Conscientiousness)',
        'color': (139, 92, 246),  # Purple #8B5CF6
        'iconName': 'conscientiousness.png',
        'description': (
            "You show strong self-discipline, goal-directed behavior, and structural organization. "
            "Your updates tend to be clear, structured, and focused on tasks, planning, achievements, "
            "and personal development."
        )
    },
    'sEXT': {
        'title': 'The Socializer (High Extraversion)',
        'color': (249, 115, 22),  # Orange #F97316
        'iconName': 'extraversion.png',
        'description': (
            "You are outgoing, energetic, and thrive on social interaction. Your writing style is "
            "expressive and warm, frequently featuring social activities, high emotional intensity, "
            "and connection-seeking statements."
        )
    },
    'sAGR': {
        'title': 'The Harmonizer (High Agreeableness)',
        'color': (16, 185, 129),  # Emerald #10B981
        'iconName': 'agreeableness.png',
        'description': (
            "You display high empathy, cooperativeness, and a compassionate attitude. Your status "
            "updates reflect politeness, supportive language, and a strong tendency to build "
            "positive, trust-based relationships."
        )
    },
    'sNEU': {
        'title': 'The Deep Feeler (High Neuroticism)',
        'color': (244, 63, 94),  # Rose #F43F5E
        'iconName': 'neuroticism.png',
        'description': (
            "You have a highly sensitive and emotionally expressive nature. Your writing is deeply "
            "personal and reflects a rich, complex inner life with strong reactions to life events, "
            "stress, or positive experiences."
        )
    }
}

# Font Helper
def getFont(fontName="arial", size=16, weight="regular"):
    """Return a PIL ImageFont, falling back to the default if no system font is found."""
    paths = []
    if fontName == "arial":
        if weight == "bold":
            paths = ["C:\\Windows\\Fonts\\arialbd.ttf", "C:\\Windows\\Fonts\\segoeuib.ttf"]
        else:
            paths = ["C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\segoeui.ttf"]

    for path in paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

    return ImageFont.load_default()

# Generate Personality Card
def generatePersonalityCard(userName, userId, scores, outputPath="outputs/user_personality_report.png"):
    imgWidth, imgHeight = 800, 1000
    img = Image.new("RGB", (imgWidth, imgHeight), color=(15, 23, 42))  # Slate 900
    draw = ImageDraw.Draw(img)
    cardBg = (30, 41, 59)  # Slate 800
    draw.rounded_rectangle(
        [(40, 40), (760, 960)],
        radius=20,
        fill=cardBg,
        outline=(51, 65, 85),  # Slate 700
        width=2
    )
    dominantTrait = max(scores, key=scores.get)
    archetype = ARCHETYPES[dominantTrait]
    iconPath = os.path.join("assets", archetype['iconName'])
    if os.path.exists(iconPath):
        try:
            icon = Image.open(iconPath)
            if icon.mode != "RGBA":
                icon = icon.convert("RGBA")
            icon = icon.resize((140, 140), Image.Resampling.LANCZOS)
            img.paste(icon, (570, 70), icon)
        except Exception as e:
            print(f"Warning: Could not load icon {iconPath}. Error: {e}")
    fontTitle = getFont("arial", 38, "bold")
    fontSubtitle = getFont("arial", 20, "regular")
    fontMeta = getFont("arial", 15, "regular")
    draw.text((80, 75), userName, fill=(255, 255, 255),  font=fontTitle)
    draw.text((80, 128), "PSYCHOMEDIA PERSONALITY PROFILE", fill=archetype['color'], font=fontSubtitle)
    draw.text((80, 165),
              f"User ID: {userId}   |   Date: {datetime.now().strftime('%Y-%m-%d')}",
              fill=(148, 163, 184), font=fontMeta)
    draw.line([(80, 230), (720, 230)], fill=(51, 65, 85), width=2)
    yStart    = 270
    rowHeight = 85
    traitLabels = {
        'sEXT': ('Extraversion', (249, 115, 22)),
        'sNEU': ('Neuroticism',  (244, 63, 94)),
        'sAGR': ('Agreeableness',  (16, 185, 129)),
        'sCON': ('Conscientiousness', (139, 92, 246)),
        'sOPN': ('Openness',  (6, 182, 212)),
    }
    fontTraitName = getFont("arial", 18, "bold")
    fontScore = getFont("arial", 18, "bold")
    for idx, (labelKey, (name, color)) in enumerate(traitLabels.items()):
        yPos = yStart + (idx * rowHeight)
        score = max(1.0, min(5.0, scores.get(labelKey, 3.0)))
        draw.text((80, yPos), name, fill=(241, 245, 249), font=fontTraitName)
        draw.text((615, yPos), f"{score:.2f} / 5.00", fill=(241, 245, 249), font=fontScore)
        barX1, barY1 = 80, yPos + 30
        barX2, barY2 = 720, yPos + 44
        draw.rounded_rectangle([(barX1, barY1), (barX2, barY2)], radius=7, fill=(51, 65, 85))
        fillWidth = int(640 * (score / 5.0))
        if fillWidth > 0:
            draw.rounded_rectangle(
                [(barX1, barY1), (barX1 + fillWidth, barY2)],
                radius=7, fill=color
            )
    yBoxStart = 710
    draw.rounded_rectangle(
        [(80, yBoxStart), (720, 900)],
        radius=12,
        fill=(15, 23, 42),
        outline=(51, 65, 85),
        width=1
    )
    fontArchTitle = getFont("arial", 20, "bold")
    draw.text(
        (105, yBoxStart + 20),
        f"Primary Archetype: {archetype['title']}",
        fill=archetype['color'], font=fontArchTitle
    )
    fontDesc = getFont("arial", 15, "regular")
    yTextPos = yBoxStart + 55
    for line in textwrap.wrap(archetype['description'], width=72):
        draw.text((105, yTextPos), line, fill=(203, 213, 225), font=fontDesc)
        yTextPos += 22
    fontFooter = getFont("arial", 12, "regular")
    draw.text(
        (220, 930),
        "PsychoMedia Model Analysis Report • Powered by Facebook Linguistic Metrics",
        fill=(100, 116, 139), font=fontFooter
    )
    os.makedirs(os.path.dirname(outputPath) or ".", exist_ok=True)
    img.save(outputPath)
    print(f"Personality profile card saved to: {outputPath}")