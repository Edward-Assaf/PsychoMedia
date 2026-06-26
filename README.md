# PsychoMedia 🧠📊

PsychoMedia is an AI-driven data science project that combines **psycholinguistics** and **social network analysis** to predict the **Big Five Personality Traits (OCEAN)** of social media users. By analyzing linguistic style, semantic characteristics of status updates, and structural properties of their social network, the system builds predictive models and outputs beautiful, personalized personality profile cards.

---

## 🚀 Key Features

*   **Psycholinguistic Feature Engineering**: Extracts over 200 semantic and emotional categories from text using the `Empath` lexicon.
*   **Stylometric Analysis**: Computes typing-style metrics such as character count, word count, exclamation/question frequency, uppercase ratio, and first-person pronoun density.
*   **Social Network Integration**: Combines text metrics with structural social network metrics (Network Size, Betweenness Centrality, Density, Brokerage, and Transitivity).
*   **Machine Learning Models**: Employs **Ridge Regression** (L2-regularized linear model) and **Random Forest Regressor** (tuned ensemble model) to predict continuous personality scores.
*   **Interactive Evaluation & Profiling**: An interactive command-line interface allows users to input their social metrics and posts, and receive real-time personality predictions.
*   **Visual Reports**: Generates stylized, premium-designed PNG reports (Personality Cards) mapping the user's primary psychological archetype and trait scores.
*   **Academic Document Builder**: Automatically compiles an 8-page academic Word document (in Arabic) documenting the data pipeline, algorithms, visualizations, and results.

---

## 📂 Project Structure

*   `psychomedia.py`: The core machine learning pipeline. Cleans raw data, engineers features, trains the Ridge Regression and Random Forest models, and saves the trained artifacts.
*   `analyze.py`: Interactive command-line interface. Guides the user through intuitive questions to capture social network metrics, accepts post text, runs prediction, and triggers personality card generation.
*   `personality_card.py`: Handles graphic generation using Pillow (`PIL`). Draws the user's personality profile card complete with custom colors, progress bars, archetypes, and text-wrapping.
*   `visualize.py`: Script to generate high-quality analysis plots (distributions, trend lines, box plots, grouped bars, and correlation heatmaps) and saves them in the `plots/` directory.
*   `datasets/`: Holds raw and preprocessed datasets (e.g., `fbDataset.csv`, `empath.csv`, scaled training/testing sets).
*   `models/`: Stores serialized models (`forest_model.joblib`, `ridge_model.joblib`), scaling matrices (`scaler.joblib`), and selected feature configurations.
*   `plots/`: Contains generated analytical charts used for data analysis and academic documentation.
*   `outputs/`: Stores generated user personality cards (e.g., `edward_report_card.png`).
*   `assets/`: Directory for holding visual assets such as icons for the different personality dimensions.

---

## 🧠 The Big Five (OCEAN) Framework

The system predicts scores on a scale of `1.0` to `5.0` for each of the five dimensions:
1.  **Openness to Experience (sOPN)**: *The Visionary* – Intellectual curiosity, creative imagination, and preference for novelty.
2.  **Conscientiousness (sCON)**: *The Organizer* – Self-discipline, organization, and goal-directed behavior.
3.  **Extraversion (sEXT)**: *The Socializer* – Social energy, outgoingness, and expressive communication style.
4.  **Agreeableness (sAGR)**: *The Harmonizer* – Compassion, cooperativeness, and warm, polite language.
5.  **Neuroticism (sNEU)**: *The Deep Feeler* – Emotional sensitivity, personal expressiveness, and inner reflection.

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/PsychoMedia.git
    cd PsychoMedia
    ```

2.  **Install Required Dependencies**:
    Make sure you have Python 3.8+ installed. Install the necessary libraries:
    ```bash
    pip install pandas numpy scikit-learn joblib empath matplotlib python-docx pillow
    ```

3.  **Ensure Datasets & Assets are in Place**:
    *   Place your raw Facebook dataset (`fbDataset.csv`) inside the `datasets/` folder.
    *   Ensure the `assets/` folder contains the required icon files (`openness.png`, `conscientiousness.png`, `extraversion.png`, `agreeableness.png`, `neuroticism.png`).

---

## 🏃 Running the Project

### 1. Train the Models
To perform data cleaning, feature extraction, train Ridge & Random Forest Regressors, and save models:
```bash
python psychomedia.py
```

### 2. Generate Data Visualizations
To output the analysis plots (distributions, correlation heatmaps, etc.) into the `plots/` directory:
```bash
python visualize.py
```

### 3. Generate Academic Word Documentation
To compile the professional academic document (Arabic Word format):
```bash
python build_documentation.py
```

### 4. Run Interactive Personality Profiler
To run the model on your own input data and generate your visual Personality Report Card:
```bash
python analyze.py
```
During the prompt, the script will ask questions about your friends count, how often you act as a bridge between friend groups, and prompts for your status updates before exporting a premium card to the `outputs/` folder.

---

## 📊 Model Evaluation Results

Models are evaluated using **Mean Squared Error (MSE)** on a 20% validation split. The Random Forest Regressor outperforms the linear model by capturing non-linear feature interactions:

| Trait Dimension | Ridge Regression (MSE) | Random Forest Regressor (MSE) |
| :--- | :---: | :---: |
| **Openness (sOPN)** | 0.32632 | **0.30197** |
| **Conscientiousness (sCON)** | 0.52934 | **0.52243** |
| **Extraversion (sEXT)** | **0.70458** | 0.72831 |
| **Agreeableness (sAGR)** | 0.44697 | **0.39729** |
| **Neuroticism (sNEU)** | 0.48253 | **0.44336** |
| **Overall Average MSE** | 0.49795 | **0.47867** |

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Contributors

*   **Batoul Mohammad Khalil**
*   **Edward Malek Assaf**
