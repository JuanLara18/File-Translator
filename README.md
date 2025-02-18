# File Translator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python script that translates columns from German to English using OpenAI's GPT API. The script supports both Excel and Stata files, maintaining the original file structure and adding new columns with translations.

## Features

- Translates specified columns from German to English
- Supports both Excel (.xlsx, .xls) and Stata (.dta) files
- Preserves original file structure and metadata
- Handles batch processing for large files
- Implements caching system for repeated phrases
- Validates translation quality with retry mechanism
- Creates detailed logs of the translation process

## Implementation Details

The script uses several key components for robust translation:

1. **Translation Cache System**
   - Identifies unique texts to avoid redundant translations
   - Maintains frequency count of repeated phrases
   - Significantly reduces API calls and processing time

2. **Batch Processing**
   - Processes texts in configurable batch sizes
   - Uses ThreadPoolExecutor for parallel translations
   - Optimizes API usage and performance

3. **Translation Validation**
   - Validates each translation's format and quality
   - Implements retry mechanism for failed translations
   - Preserves technical terms and error codes

4. **Error Handling**
   - Robust error detection and logging
   - Fallback mechanisms for failed translations
   - Detailed logging of all operations

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
- `--batch_size`: Number of rows to process at once (default: 10)

## Translation Process

1. **File Loading**
   - Reads Excel or Stata file
   - Preserves metadata and structure
   - Validates column specifications

2. **Cache Creation**
   - Creates cache of unique texts
   - Counts frequency of repeated phrases
   - Optimizes translation workload

3. **Batch Processing**
   - Processes texts in configurable batches
   - Uses parallel processing for efficiency
   - Implements robust error handling

4. **Translation and Validation**
   - Translates using OpenAI's GPT API
   - Validates translation format and quality
   - Retries failed translations individually
   - Preserves technical terms and formatting

5. **Output Generation**
   - Adds translations as new columns
   - Preserves original data structure
   - Maintains file-specific metadata

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

## Generated Files

1. **Translated File**
   - Original file with new translation columns
   - Excel: New .xlsx file with additional columns
   - Stata: New .dta file with preserved metadata

2. **Log File** (`translation.log`)
   - Translation progress and statistics
   - Error logs and handling
   - Cache performance metrics
   - Translation validation results
   - Process timing information

## Common Issues and Solutions

1. **"OpenAI API key not found"**
   - Check if `OPENAI_API_KEY.env` exists
   - Verify key is valid and not expired
   - Ensure file permissions are correct

2. **"Cannot access file"**
   - Close file in other programs
   - Verify file path and permissions
   - Check file is not locked/read-only

3. **"Column not found"**
   - Verify exact column name match
   - Check letter notation (Excel)
   - Ensure column exists in file

4. **"Translation validation failed"**
   - Check input text format
   - Verify API response format
   - Monitor retry mechanism logs

5. **"Stata metadata error"**
   - Verify Stata file version (13+)
   - Check for file corruption
   - Ensure metadata compatibility

## License

MIT License - feel free to use and modify as needed.s