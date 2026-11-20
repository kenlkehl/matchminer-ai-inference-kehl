import json
import argparse
import os
from pathlib import Path
import pandas as pd
from matchminer_ai.training.paths import no_phi
from matchminer_ai.training.trials.ctgov_metadata import extract_ctgov_metadata

def parse_clinical_trials_json(json_file_path):
    """
    Parse clinical trials JSON file into a pandas DataFrame.
    
    Args:
        json_file_path: Path to the JSON file containing clinical trial data
        
    Returns:
        pandas DataFrame with columns: nct_id, title, brief_summary, 
        eligibility_criteria, trial_text, study_type, and phases (a JSON array
        retaining ClinicalTrials.gov's structured phase values), plus ctgov
        status and dates with their original precision and actual/estimated types.
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # List to store parsed trial data
    trials = []
    
    # Parse each trial entry
    for trial in data:
        try:
            # Extract the protocol section
            protocol = trial.get('protocolSection', {})
            design = protocol.get('designModule', {})
            
            # Extract NCT ID
            nct_id = protocol.get('identificationModule', {}).get('nctId', '')
            
            # Extract title (using briefTitle, could also use officialTitle)
            title = protocol.get('identificationModule', {}).get('briefTitle', '')
            
            # Extract brief summary
            brief_summary = protocol.get('descriptionModule', {}).get('briefSummary', '')
            
            # Extract eligibility criteria
            eligibility_criteria = protocol.get('eligibilityModule', {}).get('eligibilityCriteria', '')
            
            # Create trial_text as concatenation
            trial_text = f"{title}\n{brief_summary}\n{eligibility_criteria}"
            
            # Append to list
            trials.append({
                'nct_id': nct_id,
                'study_type': design.get('studyType', ''),
                'phases': json.dumps(design.get('phases') or []),
                **extract_ctgov_metadata(protocol),
                'title': title,
                'brief_summary': brief_summary,
                'eligibility_criteria': eligibility_criteria,
                'trial_text': trial_text
            })
            
        except Exception as e:
            print(f"Error parsing trial: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(trials)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a ClinicalTrials.gov JSON export.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="Default: <data root>/no_phi/ctgov_trials.csv")
    args = parser.parse_args()
    df = parse_clinical_trials_json(args.input)
    output = Path(args.output) if args.output else no_phi("ctgov_trials.csv")
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        df.to_csv(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Parsed {len(df)} trials ({df.nct_id.nunique()} unique NCT IDs)")
    print(f"Saved {output} with study type, phases, ctgov status, and date metadata")
