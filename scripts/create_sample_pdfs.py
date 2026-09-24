"""Generate SMALL SAMPLE PDFs for local testing of the pipeline.

These are fictional, clearly-labelled demo documents (Globex operations,
Aurora Energy report, deep-learning primer). They exist so you can exercise
ingestion -> retrieval -> UI -> evaluation before adding your real 10-20 PDFs.

The script ALSO writes evaluation/questions_sample.json - 30+ real question/
answer pairs that are grounded in the EXACT pages generated here, so the
retrieval evaluation produces genuine numbers against sample data.

Usage:
    python scripts/create_sample_pdfs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf as fitz  # noqa: E402

from src.utils import ensure_dir, write_json  # noqa: E402

# ---------------------------------------------------------------------------
# Sample content: document name -> [page_1_text, page_2_text, ...]
# ---------------------------------------------------------------------------
SAMPLE_DOCS: dict[str, list[str]] = {
    "globex_operations_handbook.pdf": [
        (
            "GLOBEX OPERATIONS HANDBOOK - SAMPLE DOCUMENT (fictional, for testing)\n\n"
            "1. Company Overview\n"
            "Globex is a logistics company founded in 2012. Its headquarters are located "
            "in Rotterdam, the Netherlands. The company employs 4,200 people and operates "
            "freight and warehousing services across Europe. Globex's core business is "
            "just-in-time delivery for industrial clients."
        ),
        (
            "2. Fleet and Network\n"
            "Globex operates a fleet of 320 trucks. The network contains 14 distribution "
            "centers across eight European countries. Average on-time delivery is 96 percent. "
            "Each distribution center is connected to at least two major motorways. Trucks are "
            "tracked with GPS and drivers use an internal routing application."
        ),
        (
            "3. Warehouse Safety\n"
            "All warehouse staff must wear PPE: hard hats, steel-toe boots, and high-visibility "
            "vests. Forklift operators must hold a valid certification. A safety drill is held "
            "every month. Spills must be reported to the shift supervisor immediately. Visitors "
            "receive a short safety briefing before entering the floor."
        ),
        (
            "4. Sustainability\n"
            "Globex targets 40 percent of its fleet to be electric by 2030. Distribution centers "
            "run partly on solar panels. The company also measures emissions per shipment. "
            "Internal audits review progress every quarter and publish the results annually."
        ),
        (
            "5. Customer Service Standards\n"
            "The customer service team replies within 24 hours to every request. Claim "
            "resolution rate is 99.2 percent, with claims closed within 14 days. High-priority "
            "shipments get real-time tracking updates. Customers can reach support by phone, "
            "email, or a web portal."
        ),
        (
            "6. Hiring and Training\n"
            "Globex runs two hiring intakes per year. Referral bonus is 1,500 EUR. New drivers "
            "complete a six-week training program. Warehouse trainees shadow a mentor for the "
            "first month. Performance reviews happen twice a year."
        ),
    ],
    "aurora_energy_report_2025.pdf": [
        (
            "AURORA ENERGY ANNUAL REPORT 2025 - SAMPLE DOCUMENT (fictional, for testing)\n\n"
            "Overview\n"
            "Aurora Energy is a utility company headquartered in Oslo, Norway. In 2025 the "
            "company reported revenue of 2.4 billion EUR. Aurora supplies electricity and "
            "heating to residential and business customers in the Nordic region."
        ),
        (
            "Renewable Generation\n"
            "Aurora's installed capacity consists of 850 MW of wind, 320 MW of solar, and "
            "1,100 MW of hydro power. Hydropower remains the backbone of the portfolio, "
            "delivering roughly half of the annual output."
        ),
        (
            "Customers\n"
            "Aurora serves 1.9 million households. About 62 percent of customers renew their "
            "subscriptions each year. Average monthly churn is under five percent. The customer "
            "app is used by more than 700,000 people."
        ),
        (
            "Grid Investment\n"
            "Aurora is investing 310 million EUR in smart meters. The rollout covers 40 percent "
            "of the service territory by 2027. Smart metering improves outage detection and "
            "enables time-of-use pricing for households."
        ),
        (
            "Emissions\n"
            "Aurora reduced emissions by 18 percent compared with 2020 levels. The company "
            "targets a 55 percent reduction by 2030. Renewable power purchases apply to all "
            "business electricity products."
        ),
        (
            "Workforce\n"
            "Aurora employs 1,750 people. Women make up 31 percent of engineering roles. The "
            "company awards 60 scholarships per year in energy technology. All employees receive "
            "five paid volunteering days annually."
        ),
    ],
    "deep_learning_primer.pdf": [
        (
            "DEEP LEARNING PRIMER - SAMPLE DOCUMENT (educational, for testing)\n\n"
            "What is Deep Learning?\n"
            "Deep learning is a subset of machine learning. It uses neural networks with many "
            "layers to learn patterns from data. Training optimizes a loss function using "
            "gradient descent. Deep learning powers image recognition, speech, and language "
            "applications."
        ),
        (
            "Neural Networks\n"
            "A neural network consists of layers of neurons. Each neuron has weights and a "
            "bias. Activation functions such as ReLU and sigmoid introduce nonlinearity. "
            "Common architectures include dense layers, convolutional layers, and recurrent "
            "layers."
        ),
        (
            "Training\n"
            "Networks are trained with backpropagation. Data is processed in batches over many "
            "epochs. The learning rate controls step size during gradient descent. Overfitting "
            "occurs when the model memorizes the training data."
        ),
        (
            "Regularization\n"
            "Regularization methods reduce overfitting. Dropout randomly drops neurons during "
            "training. L2 regularization penalizes large weights. Early stopping ends training "
            "when validation performance stops improving."
        ),
        (
            "Convolutional Networks\n"
            "Convolutional neural networks (CNNs) are designed for images. They use kernels to "
            "detect features and pooling layers to reduce spatial size. CNNs achieve strong "
            "results on object recognition tasks."
        ),
        (
            "Transformers\n"
            "Transformers rely on the attention mechanism. Attention weighs how much each token "
            "relates to every other token. Transformer models form the basis of modern large "
            "language models. They process sequences in parallel, which accelerates training."
        ),
    ],
}

# ---------------------------------------------------------------------------
# Sample questions: (question, answer, [(doc, page), ...])
# Page numbers refer to the pages generated above (1-based).
# ---------------------------------------------------------------------------
SAMPLE_QAS: list[dict] = [
    # --- Globex -------------------------------------------------------------
    {"question": "When was Globex founded and where is its headquarters?",
     "answer": "Globex was founded in 2012 and its headquarters are in Rotterdam, the Netherlands.",
     "sources": [("globex_operations_handbook.pdf", 1)]},
    {"question": "How many people does Globex employ?",
     "answer": "Globex employs 4,200 people.", "sources": [("globex_operations_handbook.pdf", 1)]},
    {"question": "How many trucks does Globex operate?",
     "answer": "Globex operates a fleet of 320 trucks.", "sources": [("globex_operations_handbook.pdf", 2)]},
    {"question": "What is Globex's average on-time delivery rate?",
     "answer": "The average on-time delivery rate is 96 percent.",
     "sources": [("globex_operations_handbook.pdf", 2)]},
    {"question": "How many distribution centers does Globex have?",
     "answer": "Globex has 14 distribution centers across eight European countries.",
     "sources": [("globex_operations_handbook.pdf", 2)]},
    {"question": "What PPE must warehouse staff wear at Globex?",
     "answer": "Hard hats, steel-toe boots, and high-visibility vests.",
     "sources": [("globex_operations_handbook.pdf", 3)]},
    {"question": "How often are safety drills held at Globex warehouses?",
     "answer": "A safety drill is held every month.", "sources": [("globex_operations_handbook.pdf", 3)]},
    {"question": "What is Globex's target share of electric trucks by 2030?",
     "answer": "Globex targets 40 percent of its fleet to be electric by 2030.",
     "sources": [("globex_operations_handbook.pdf", 4)]},
    {"question": "What renewable source do Globex distribution centers use?",
     "answer": "Distribution centers run partly on solar panels.",
     "sources": [("globex_operations_handbook.pdf", 4)]},
    {"question": "How quickly does Globex customer service reply?",
     "answer": "The team replies within 24 hours to every request.",
     "sources": [("globex_operations_handbook.pdf", 5)]},
    {"question": "What is Globex's claim resolution rate and window?",
     "answer": "99.2 percent of claims are resolved within 14 days.",
     "sources": [("globex_operations_handbook.pdf", 5)]},
    {"question": "How long is Globex's new-driver training program?",
     "answer": "New drivers complete a six-week training program.",
     "sources": [("globex_operations_handbook.pdf", 6)]},
    {"question": "What is the referral bonus at Globex?",
     "answer": "The referral bonus is 1,500 EUR.", "sources": [("globex_operations_handbook.pdf", 6)]},
    {"question": "How many hiring intakes does Globex run per year?",
     "answer": "Globex runs two hiring intakes per year.",
     "sources": [("globex_operations_handbook.pdf", 6)]},
    # --- Aurora -------------------------------------------------------------
    {"question": "What revenue did Aurora Energy report for 2025?",
     "answer": "Aurora reported revenue of 2.4 billion EUR in 2025.",
     "sources": [("aurora_energy_report_2025.pdf", 1)]},
    {"question": "Where is Aurora Energy headquartered?",
     "answer": "Aurora Energy is headquartered in Oslo, Norway.",
     "sources": [("aurora_energy_report_2025.pdf", 1)]},
    {"question": "What is Aurora's wind generation capacity?",
     "answer": "Aurora has 850 MW of installed wind capacity.",
     "sources": [("aurora_energy_report_2025.pdf", 2)]},
    {"question": "What is Aurora's hydro power capacity?",
     "answer": "Aurora has 1,100 MW of installed hydro capacity.",
     "sources": [("aurora_energy_report_2025.pdf", 2)]},
    {"question": "How many households does Aurora Energy serve?",
     "answer": "Aurora serves 1.9 million households.",
     "sources": [("aurora_energy_report_2025.pdf", 3)]},
    {"question": "What share of Aurora customers renew their subscriptions each year?",
     "answer": "About 62 percent of customers renew each year.",
     "sources": [("aurora_energy_report_2025.pdf", 3)]},
    {"question": "How much is Aurora investing in smart meters?",
     "answer": "Aurora is investing 310 million EUR in smart meters.",
     "sources": [("aurora_energy_report_2025.pdf", 4)]},
    {"question": "What share of the service territory will have smart meters by 2027?",
     "answer": "The rollout covers 40 percent of the territory by 2027.",
     "sources": [("aurora_energy_report_2025.pdf", 4)]},
    {"question": "What is Aurora's 2030 emissions reduction target?",
     "answer": "Aurora targets a 55 percent emissions reduction by 2030 compared with 2020.",
     "sources": [("aurora_energy_report_2025.pdf", 5)]},
    {"question": "What emissions reduction has Aurora achieved compared with 2020?",
     "answer": "Aurora reduced emissions by 18 percent compared with 2020.",
     "sources": [("aurora_energy_report_2025.pdf", 5)]},
    {"question": "How many people does Aurora Energy employ?",
     "answer": "Aurora employs 1,750 people.", "sources": [("aurora_energy_report_2025.pdf", 6)]},
    {"question": "How many scholarships does Aurora award each year?",
     "answer": "Aurora awards 60 scholarships per year in energy technology.",
     "sources": [("aurora_energy_report_2025.pdf", 6)]},
    # --- Deep learning primer ----------------------------------------------
    {"question": "What is deep learning a subset of?",
     "answer": "Deep learning is a subset of machine learning.",
     "sources": [("deep_learning_primer.pdf", 1)]},
    {"question": "Name two activation functions used in neural networks.",
     "answer": "ReLU and sigmoid are example activation functions.",
     "sources": [("deep_learning_primer.pdf", 2)]},
    {"question": "What algorithm trains networks by propagating error backwards?",
     "answer": "Networks are trained with backpropagation.",
     "sources": [("deep_learning_primer.pdf", 3)]},
    {"question": "What does the learning rate control?",
     "answer": "The learning rate controls step size during gradient descent.",
     "sources": [("deep_learning_primer.pdf", 3)]},
    {"question": "List three regularization methods that reduce overfitting.",
     "answer": "Dropout, L2 regularization, and early stopping.",
     "sources": [("deep_learning_primer.pdf", 4)]},
    {"question": "What network type uses kernels and pooling for images?",
     "answer": "Convolutional neural networks (CNNs) use kernels and pooling.",
     "sources": [("deep_learning_primer.pdf", 5)]},
    {"question": "What mechanism do transformers rely on?",
     "answer": "Transformers rely on the attention mechanism.",
     "sources": [("deep_learning_primer.pdf", 6)]},
    # --- Outside scope (no expected sources) -------------------------------
    {"question": "What is the current share price of Globex?",
     "answer": "Not covered by the provided documents - the correct answer is 'I don't know based on the provided documents.'",
     "sources": []},
]


def render_pdf(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)  # A4
        rect = fitz.Rect(50, 50, 545, 800)
        page.insert_textbox(rect, text, fontsize=10, align=fitz.TEXT_ALIGN_LEFT)
    doc.save(str(path))
    doc.close()


def build_questions() -> list[dict]:
    items = []
    for i, qa in enumerate(SAMPLE_QAS, start=1):
        items.append({
            "id": i,
            "question": qa["question"],
            "expected_answer": qa["answer"],
            "expected_sources": [
                {"document": doc, "page": page} for doc, page in qa["sources"]
            ],
        })
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sample PDFs + questions for testing.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "documents")
    parser.add_argument("--questions-output", type=Path, default=PROJECT_ROOT / "evaluation" / "questions_sample.json")
    args = parser.parse_args()

    ensure_dir(args.output)
    for name, pages in SAMPLE_DOCS.items():
        render_pdf(args.output / name, pages)
        print(f"created {args.output / name} ({len(pages)} pages)")

    questions = build_questions()
    write_json(args.questions_output, questions)
    print(f"created {args.questions_output} ({len(questions)} questions)")
    print("\nNOTE: These are fictional sample documents for local testing.")
    print("Replace them with your real 10-20 PDFs before submission.")


if __name__ == "__main__":
    main()