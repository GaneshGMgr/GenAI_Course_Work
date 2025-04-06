import json
import glob

print("Starting notebook cleanup...")
fixed_count = 0

for notebook_path in glob.glob("**/*.ipynb", recursive=True):
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        needs_fix = False
        if 'metadata' in nb and 'widgets' in nb['metadata']:
            for widget_id, widget_data in nb['metadata']['widgets'].items():
                if 'state' not in widget_data:
                    widget_data['state'] = {}
                    needs_fix = True
        
        if needs_fix:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)
            print(f"Fixed: {notebook_path}")
            fixed_count += 1
            
    except Exception as e:
        print(f"Error in {notebook_path}: {str(e)}")

print(f"\nFinished. Fixed {fixed_count} notebooks.")