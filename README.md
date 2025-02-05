# Excel Translator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python script that translates Excel columns from German to English using OpenAI's GPT API. The script maintains the original Excel structure and adds a new column with translations.

## Features

- Translates specified Excel column from German to English
- Preserves original Excel file structure
- Handles batch processing for large files
- Caches repeated phrases for efficiency
- Creates detailed logs of the translation process

## Setup

### 1. Install Python Requirements

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

1. Get your API key from [OpenAI's platform](https://platform.openai.com/api-keys)
2. Create a file named `OPENAI_API_KEY.env` in the project folder
3. Add your API key to the file:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Basic Command
```bash
python excel_translation.py --input your_file.xlsx --column Column_Name
```

### All Options
```bash
python excel_translation.py 
    --input input_file.xlsx 
    --output translated_file.xlsx 
    --column Column_Name 
    --new_col "Translation" 
    --batch_size 10
```

### Arguments

- `--input`: Your Excel file path (required)
- `--output`: Where to save the translated file (optional)
- `--column`: Column to translate - name or letter (required)
- `--new_col`: Name for the translation column (default: "Translation")
- `--batch_size`: Rows to process at once (default: 10)

## Examples

### Product Descriptions

**Input:**
```
| ProductID | Description_DE                         |
|-----------|---------------------------------------|
| 1         | Bluetooth Kopfhörer mit Geräuschunterdrückung |
| 2         | Wasserdichte Sportuhr mit Herzfrequenzmesser |
```

**Output:**
```
| ProductID | Description_DE                         | Translation                               |
|-----------|----------------------------------------|-------------------------------------------|
| 1         | Bluetooth Kopfhörer mit Geräuschunterdrückung | Bluetooth headphones with noise cancellation |
| 2         | Wasserdichte Sportuhr mit Herzfrequenzmesser | Waterproof sports watch with heart rate monitor |
```

### Customer Reviews

**Input:**
```
| ReviewID | Comments                               |
|----------|----------------------------------------|
| 1        | Sehr gutes Produkt, schnelle Lieferung |
| 2        | Die Qualität könnte besser sein        |
```

**Output:**
```
| ReviewID | Comments                               | Translation                          |
|----------|----------------------------------------|--------------------------------------|
| 1        | Sehr gutes Produkt, schnelle Lieferung | Very good product, fast delivery    |
| 2        | Die Qualität könnte besser sein        | The quality could be better         |
```

## Files Generated

- **Translated Excel File**: Original file with new translation column
- **Log File** (`excel_translation.log`): Contains:
  - Translation progress
  - Any errors encountered
  - Processing statistics
  - Translation completion time

## Common Issues

1. **"OpenAI API key not found"**
   - Check if `OPENAI_API_KEY.env` exists and contains your key
   - Verify the key is valid and not expired

2. **"Cannot access Excel file"**
   - Make sure the Excel file is not open in another program
   - Verify file path is correct

3. **"Column not found"**
   - Confirm the column name matches exactly
   - If using letter notation, ensure the column exists

## License

MIT License - feel free to use and modify as needed.