#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Translate regular_summary and brief_summary to Chinese and Japanese using vLLM with OpenAI API.

This script:
1. Reads all JSONL files from an input folder
2. For records with is_truncated=false, translates regular_summary and brief_summary to both Chinese and Japanese
3. Outputs new JSONL files with added fields:
   - regular_summary_cn, brief_summary_cn (Chinese)
   - regular_summary_ja, brief_summary_ja (Japanese)
"""

import os
import json
import argparse
import asyncio
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from openai import AsyncOpenAI
from json_repair import repair_json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate summaries in JSONL files to Chinese and Japanese using vLLM"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing JSONL files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for translated JSONL files"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for vLLM API endpoint (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to use for translation"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for authentication (default: EMPTY for local vLLM)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for generation (default: 0.3, lower for more accurate translation)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens for generation (default: 8192)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent API requests (default: 10)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout for API requests in seconds (default: 180)"
    )
    return parser.parse_args()


def build_translation_prompt(regular_summary: Optional[str], brief_summary: Optional[str]) -> str:
    """
    Build a detailed prompt for translating summaries to both Chinese and Japanese.
    """
    prompt = """You are a professional translator specializing in English to Chinese and Japanese translation. Your task is to accurately translate image descriptions.

## Translation Requirements:

### For Chinese (中文):
1. **Accuracy**: Faithfully convey the original meaning without omitting details or adding content
2. **Terminology**: Use official Chinese names for anime/game characters; use professional art terms (构图, 光影, 色调, etc.)
3. **Fluency**: Ensure natural Chinese expression

### For Japanese (日本語):
1. **Accuracy**: 原文の意味に忠実に、詳細を漏らさない
2. **Terminology**: アニメ/ゲームキャラクターは公式日本語名を使用；専門的な芸術用語を使用
3. **Fluency**: 自然な日本語表現を心がける

## Content to Translate:

"""
    
    has_content = False
    
    if regular_summary and str(regular_summary).strip() and str(regular_summary).lower() != 'nan':
        prompt += f"### Regular Summary (Detailed Description):\n{regular_summary}\n\n"
        has_content = True
    
    if brief_summary and str(brief_summary).strip() and str(brief_summary).lower() != 'nan':
        prompt += f"### Brief Summary:\n{brief_summary}\n\n"
        has_content = True
    
    if not has_content:
        return None
    
    prompt += """## Output Format:
Please output the translation results in the following JSON format. Do not output anything else:

```json
{
    "regular_summary_cn": "Chinese translation of detailed description (null if no original)",
    "brief_summary_cn": "Chinese translation of brief summary (null if no original)",
    "regular_summary_ja": "Japanese translation of detailed description (null if no original)",
    "brief_summary_ja": "Japanese translation of brief summary (null if no original)"
}
```

Please translate now:"""
    
    return prompt


def parse_translation_response(response_text: str) -> dict:
    """
    Parse the translation response from the model using json_repair.
    
    Args:
        response_text: The raw response text from the model
    
    Returns:
        A dictionary with all translation fields
    """
    import re
    
    default_result = {
        "regular_summary_cn": None,
        "brief_summary_cn": None,
        "regular_summary_ja": None,
        "brief_summary_ja": None
    }
    
    # Try to extract JSON from markdown code fence first
    json_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(json_pattern, response_text)
    
    # Candidate texts to try parsing
    candidates = matches + [response_text]
    
    for candidate in candidates:
        try:
            # Use json_repair to fix and parse potentially malformed JSON
            repaired = repair_json(candidate.strip(), return_objects=True)
            
            if isinstance(repaired, dict):
                result = default_result.copy()
                for key in default_result.keys():
                    if key in repaired:
                        result[key] = repaired[key]
                return result
        except Exception:
            continue
    
    # If json_repair fails, try to find JSON-like structure and repair it
    json_obj_pattern = r'\{[\s\S]*?"regular_summary_cn"[\s\S]*?\}'
    matches = re.findall(json_obj_pattern, response_text)
    
    for match in matches:
        try:
            repaired = repair_json(match, return_objects=True)
            if isinstance(repaired, dict):
                result = default_result.copy()
                for key in default_result.keys():
                    if key in repaired:
                        result[key] = repaired[key]
                return result
        except Exception:
            continue
    
    # If all parsing fails, return default None values
    return default_result


async def translate_single_record(
    client: AsyncOpenAI,
    record: dict,
    model_name: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore
) -> dict:
    """
    Translate a single record's summaries to both Chinese and Japanese.
    
    Args:
        client: AsyncOpenAI client instance
        record: The original record dictionary
        model_name: Model name to use
        temperature: Generation temperature
        max_tokens: Maximum tokens for generation
        semaphore: Semaphore for controlling concurrency
    
    Returns:
        The record with added translation fields
    """
    # Create a copy of the record
    result = record.copy()
    
    # Check if translation is needed
    is_truncated = record.get("is_truncated", True)
    
    if is_truncated:
        # Skip truncated records, just add None fields
        result["regular_summary_cn"] = None
        result["brief_summary_cn"] = None
        result["regular_summary_ja"] = None
        result["brief_summary_ja"] = None
        return result
    
    regular_summary = record.get("regular_summary")
    brief_summary = record.get("brief_summary")
    
    # Build the prompt
    prompt = build_translation_prompt(regular_summary, brief_summary)
    
    if prompt is None:
        # No content to translate
        result["regular_summary_cn"] = None
        result["brief_summary_cn"] = None
        result["regular_summary_ja"] = None
        result["brief_summary_ja"] = None
        return result
    
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator. Translate the given image descriptions accurately to both Chinese and Japanese. Output only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            response_text = response.choices[0].message.content
            translation = parse_translation_response(response_text)
            
            result["regular_summary_cn"] = translation["regular_summary_cn"]
            result["brief_summary_cn"] = translation["brief_summary_cn"]
            result["regular_summary_ja"] = translation["regular_summary_ja"]
            result["brief_summary_ja"] = translation["brief_summary_ja"]
            
        except Exception as e:
            print(f"\nError translating record: {e}")
            result["regular_summary_cn"] = None
            result["brief_summary_cn"] = None
            result["regular_summary_ja"] = None
            result["brief_summary_ja"] = None
    
    return result


async def process_single_file(
    client: AsyncOpenAI,
    input_path: Path,
    output_path: Path,
    model_name: str,
    temperature: float,
    max_tokens: int,
    concurrency: int
) -> tuple[int, int, int]:
    """
    Process a single JSONL file.
    """
    # Read all records from input file
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"\nWarning: Failed to parse line in {input_path}: {e}")
                    continue
    
    if not records:
        print(f"\nWarning: No valid records found in {input_path}")
        return 0, 0, 0
    
    # Create semaphore for controlling concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    # Count records to translate
    to_translate = sum(1 for r in records if not r.get("is_truncated", True))
    
    print(f"\n📄 Processing: {input_path.name}")
    print(f"   Total records: {len(records)} | To translate: {to_translate}")
    
    # Process all records with progress bar
    with tqdm(total=len(records), desc="   Translating (CN+JA)", unit="records") as pbar:
        async def process_with_update(record):
            result = await translate_single_record(
                client, record, model_name, temperature, max_tokens, semaphore
            )
            pbar.update(1)
            return result
        
        tasks = [process_with_update(record) for record in records]
        results = await asyncio.gather(*tasks)
    
    # Write results to output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    translated = sum(
        1 for r in results 
        if r.get("regular_summary_cn") or r.get("brief_summary_cn") 
        or r.get("regular_summary_ja") or r.get("brief_summary_ja")
    )
    skipped = len(records) - to_translate
    
    print(f"   ✅ Completed: {translated} translated, {skipped} skipped (truncated)")
    
    return len(records), translated, skipped


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSONL files
    jsonl_files = list(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"Error: No JSONL files found in '{input_dir}'")
        return
    
    print("=" * 60)
    print("🚀 Summary Translation Tool (Chinese + Japanese)")
    print("=" * 60)
    print(f"📁 Input directory:  {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔗 API Base URL:     {args.base_url}")
    print(f"🤖 Model:            {args.model_name}")
    print(f"🌡️  Temperature:      {args.temperature}")
    print(f"📝 Max tokens:       {args.max_tokens}")
    print(f"⚡ Concurrency:      {args.concurrency}")
    print(f"📄 Files to process: {len(jsonl_files)}")
    print("=" * 60)
    
    # Initialize OpenAI client
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout
    )
    
    # Process each file
    total_records = 0
    total_translated = 0
    total_skipped = 0
    
    for jsonl_file in jsonl_files:
        output_path = output_dir / jsonl_file.name
        
        try:
            records, translated, skipped = await process_single_file(
                client=client,
                input_path=jsonl_file,
                output_path=output_path,
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency
            )
            
            total_records += records
            total_translated += translated
            total_skipped += skipped
            
        except Exception as e:
            print(f"\n❌ Error processing {jsonl_file.name}: {e}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"Total records processed: {total_records}")
    print(f"Successfully translated: {total_translated}")
    print(f"Skipped (truncated):     {total_skipped}")
    print(f"\nOutput fields added:")
    print(f"  - regular_summary_cn (Chinese)")
    print(f"  - brief_summary_cn (Chinese)")
    print(f"  - regular_summary_ja (Japanese)")
    print(f"  - brief_summary_ja (Japanese)")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
