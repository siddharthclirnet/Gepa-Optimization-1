import pandas as pd
import dspy
from dspy.evaluate import Evaluate
from gemma_adapter import GemmaAdapter

# Configure DSPy with Gemma
lm = dspy.LM(
    model="gemini/gemma-3-27b-it",
    api_key="AIzaSyAuAsuPM3l0zA40tWfonxJdoyEOdrUWbk0",
    cache=False,
)
dspy.settings.configure(lm=lm, adapter=GemmaAdapter())

class PANCardExtraction(dspy.Signature):
    image_url = dspy.InputField(desc="PAN card image URL")
    name = dspy.OutputField(desc="Full name of cardholder")
    date_of_birth = dspy.OutputField(desc="Date of birth in YYYY-MM-DD format")
    id_number = dspy.OutputField(desc="PAN number (5 letters, 4 numbers, 1 letter)")
    father_name = dspy.OutputField(desc="Father's name of cardholder")
    has_income_tax_logo = dspy.OutputField(desc="TRUE if Income Tax Department logo visible")
    has_gov_logo = dspy.OutputField(desc="TRUE if Govt. of India logo visible")
    has_photo = dspy.OutputField(desc="TRUE if cardholder photo visible")
    has_hologram_qrcode = dspy.OutputField(desc="TRUE if hologram or QR code visible")
    has_national_emblem = dspy.OutputField(desc="TRUE if national emblem visible")
    has_signature = dspy.OutputField(desc="TRUE if signature visible")
    is_valid_pan = dspy.OutputField(desc="1 if PAN is valid, 0 if invalid")

class GemmaPANExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(PANCardExtraction)

    def forward(self, image_url: str):
        return self.extract(image_url=image_url)

def _match(pred_value, gold_value):
    """Helper function for field matching"""
    if pred_value is None and gold_value is None:
        return 1
    if pred_value is None or gold_value is None:
        return 0
    
    pred_str = str(pred_value).strip().lower()
    gold_str = str(gold_value).strip().lower()
    
    # For boolean fields
    if gold_str in ['true', '1', 'yes'] and pred_str in ['true', '1', 'yes']:
        return 1
    if gold_str in ['false', '0', 'no'] and pred_str in ['false', '0', 'no']:
        return 1
    
    # For text fields
    if pred_str == gold_str:
        return 1
    
    # Flexible matching for names
    if gold_str in pred_str or pred_str in gold_str:
        return 1
    
    return 0

def metric_with_feedback(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Metric function that returns score between 0-1"""
    # Check each field
    ok_name = _match(pred.name, gold.name)
    ok_dob = _match(pred.date_of_birth, gold.date_of_birth)
    ok_id = _match(pred.id_number, gold.id_number)
    ok_fathername = _match(pred.father_name, gold.father_name)
    ok_incometax = _match(pred.has_income_tax_logo, gold.has_income_tax_logo)
    ok_govlogo = _match(pred.has_gov_logo, gold.has_gov_logo)
    ok_photo = _match(pred.has_photo, gold.has_photo)
    ok_qr = _match(pred.has_hologram_qrcode, gold.has_hologram_qrcode)
    ok_emblem = _match(pred.has_national_emblem, gold.has_national_emblem)
    ok_signature = _match(pred.has_signature, gold.has_signature)
    ok_validity = _match(str(pred.is_valid_pan), str(gold.is_valid_pan))
    
    # Compute score (average of correct fields)
    score = (ok_name + ok_dob + ok_id + ok_fathername + ok_incometax + 
             ok_govlogo + ok_photo + ok_qr + ok_emblem + ok_signature + ok_validity) / 11.0
    
    return score

def load_complete_dataset():
    """Load all rows from the Excel dataset"""
    print("📊 Loading complete PAN card dataset...")
    
    try:
        # Load Excel file
        df = pd.read_excel(r"D:\Clirnet Internship Work\running_dspy_with_gemma27b_image_input\data\PAN_CARD_Datasets.xlsx")
        df.columns = df.columns.str.strip()

        # Rename columns to match expected format
        df = df.rename(
            columns={
                "Name": "name",
                "DOB": "date_of_birth",
                "PAN Number": "id_number",
                "url": "image_url",
                "Father Name": "father_name",
                "IT Logo": "has_income_tax_logo",
                "Gov Logo": "has_gov_logo",
                "Photo": "has_photo",
                "Hologram_QR": "has_hologram_qrcode",
                "National Emblem": "has_national_emblem",
                "Signature": "has_signature",
                "is_valid": "is_valid_pan",
            }
        )

        # Format date of birth
        df["date_of_birth"] = pd.to_datetime(
            df["date_of_birth"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

        # Create dspy.Example objects for all rows
        examples = []
        for _, row in df.iterrows():
            try:
                ex = dspy.Example(
                    image_url=str(row["image_url"]),
                    name=str(row["name"]) if pd.notna(row["name"]) else "",
                    date_of_birth=str(row["date_of_birth"]) if pd.notna(row["date_of_birth"]) else "",
                    id_number=str(row["id_number"]) if pd.notna(row["id_number"]) else "",
                    father_name=str(row["father_name"]) if pd.notna(row["father_name"]) else "",
                    has_income_tax_logo=bool(row["has_income_tax_logo"]),
                    has_gov_logo=bool(row["has_gov_logo"]),
                    has_photo=bool(row["has_photo"]),
                    has_hologram_qrcode=bool(row["has_hologram_qrcode"]),
                    has_national_emblem=bool(row["has_national_emblem"]),
                    has_signature=bool(row["has_signature"]),
                    is_valid_pan=bool(row["is_valid_pan"]),
                ).with_inputs("image_url")
                examples.append(ex)
            except Exception as e:
                print(f"⚠️  Warning: Skipping row due to error: {e}")
                continue

        print(f"✅ Successfully loaded {len(examples)} examples from dataset")
        return examples
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []

def evaluate_all_pan_cards():
    """Evaluate all PAN cards using DSPy Evaluate with individual and summarized results"""
    print("🚀 PAN CARD EVALUATION USING DSPY EVALUATE")
    print("=" * 70)
    
    # Load all examples from dataset
    all_examples = load_complete_dataset()
    
    if not all_examples:
        print("❌ No data loaded. Exiting.")
        return
    
    # Initialize the extractor
    extractor = GemmaPANExtractor()
    
    print(f"🎯 Evaluating {len(all_examples)} PAN cards...")
    
    # Use DSPy Evaluate for all examples
    evaluator = Evaluate(
        devset=all_examples,
        metric=metric_with_feedback,
        display_progress=True,
        display_table=len(all_examples),
        num_threads=1
    )
    
    # Run the evaluation
    evaluation_result = evaluator(extractor)
    
    # Display summarized results at the end
    print("\n" + "=" * 70)
    print("📊 SUMMARIZED RESULTS - ALL FIELDS")
    print("=" * 70)
    
    # Calculate field-wise accuracy
    field_stats = calculate_field_accuracy(extractor, all_examples)
    
    print(f"Overall Accuracy: {evaluation_result.score:.3f}/1.000")
    print(f"Total PAN Cards Evaluated: {len(all_examples)}")
    
    print(f"\nField-wise Accuracy:")
    print("-" * 40)
    for field, accuracy in field_stats.items():
        print(f"{field:25}: {accuracy:.3f}")
    
    return evaluation_result

def calculate_field_accuracy(extractor, examples):
    """Calculate accuracy for each field"""
    field_correct = {field: 0 for field in [
        'name', 'date_of_birth', 'id_number', 'father_name', 
        'has_income_tax_logo', 'has_gov_logo', 'has_photo',
        'has_hologram_qrcode', 'has_national_emblem', 'has_signature', 'is_valid_pan'
    ]}
    
    total_examples = len(examples)
    
    for example in examples:
        try:
            prediction = extractor(image_url=example.image_url)
            
            # Check each field
            field_correct['name'] += _match(prediction.name, example.name)
            field_correct['date_of_birth'] += _match(prediction.date_of_birth, example.date_of_birth)
            field_correct['id_number'] += _match(prediction.id_number, example.id_number)
            field_correct['father_name'] += _match(prediction.father_name, example.father_name)
            field_correct['has_income_tax_logo'] += _match(prediction.has_income_tax_logo, example.has_income_tax_logo)
            field_correct['has_gov_logo'] += _match(prediction.has_gov_logo, example.has_gov_logo)
            field_correct['has_photo'] += _match(prediction.has_photo, example.has_photo)
            field_correct['has_hologram_qrcode'] += _match(prediction.has_hologram_qrcode, example.has_hologram_qrcode)
            field_correct['has_national_emblem'] += _match(prediction.has_national_emblem, example.has_national_emblem)
            field_correct['has_signature'] += _match(prediction.has_signature, example.has_signature)
            field_correct['is_valid_pan'] += _match(str(prediction.is_valid_pan), str(example.is_valid_pan))
            
        except Exception as e:
            print(f"⚠️  Warning: Skipping example due to error: {e}")
            continue
    
    # Calculate accuracy for each field
    field_accuracy = {}
    for field, correct_count in field_correct.items():
        accuracy = correct_count / total_examples if total_examples > 0 else 0
        field_accuracy[field] = accuracy
    
    return field_accuracy

# Run the evaluation
if __name__ == "__main__":
    result = evaluate_all_pan_cards()