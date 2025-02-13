# File Translator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python script that translates columns from German to English using OpenAI's GPT API. The script supports both Excel and Stata files, maintaining the original file structure and adding new columns with translations.

## Features

- Translates specified columns from German to English
- Supports both Excel (.xlsx, .xls) and Stata (.dta) files
- Preserves original file structure and metadata
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
python translation.py --input your_file.[xlsx|dta] --column Column_Name
```

### All Options
```bash
python translation.py 
    --input input_file.[xlsx|dta] 
    --output translated_file.[xlsx|dta] 
    --column Column_Name 
    --new_col "Translation" 
    --batch_size 10
```

### Arguments

- `--input`: Your file path (Excel or Stata) (required)
- `--output`: Where to save the translated file (optional)
- `--column`: Column to translate - name or letter (required)
- `--new_col`: Name for the translation column (default: "Translation")
- `--batch_size`: Rows to process at once (default: 10)

## Examples

### Excel: Product Descriptions

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

### Stata: Survey Responses

The script maintains Stata-specific features:
- Variable labels
- Value labels
- Data types
- Metadata preservation

For example:
```stata
* Input Stata file
label variable comments "Kundenkommentare"
label define status 1 "Sehr zufrieden" 2 "Zufrieden" 3 "Unzufrieden"

* After translation
label variable comments "Customer comments"
label define status 1 "Very satisfied" 2 "Satisfied" 3 "Unsatisfied"
```

## Files Generated

- **Translated File**: Original file with new translation columns
  - Excel: New .xlsx file with additional columns
  - Stata: New .dta file with preserved metadata and labels
- **Log File** (`translation.log`): Contains:
  - Translation progress
  - Any errors encountered
  - Processing statistics
  - Translation completion time

## Common Issues

1. **"OpenAI API key not found"**
   - Check if `OPENAI_API_KEY.env` exists and contains your key
   - Verify the key is valid and not expired

2. **"Cannot access file"**
   - Make sure the file is not open in another program
   - Verify file path is correct

3. **"Column not found"**
   - Confirm the column name matches exactly
   - If using letter notation (Excel), ensure the column exists

4. **"Stata metadata error"**
   - Check if the Stata file version is compatible (Stata 13+)
   - Ensure the file is not corrupted

## License

MIT License - feel free to use and modify as needed.