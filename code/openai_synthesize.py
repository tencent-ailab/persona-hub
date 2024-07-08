import argparse
import json
from openai import OpenAI
from prompt_templates import instruction_template, knowledge_template, npc_template, math_template
from datasets import load_dataset
from tqdm import tqdm

system_prompt = '''You are a helpful assistant.'''
client = OpenAI()   # set up your config/env/api for calling openai models

def get_response(user_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {"role": "system", "content":  f"{system_prompt}"},
            {"role": "user", "content": f"{user_prompt}"}
        ]
    )
    return completion.choices[0].message.content

def main(args):
    # Load the appropriate template
    if args.template == "instruction":
        template = instruction_template
    elif args.template == "knowledge":
        template = knowledge_template
    elif args.template == "npc":
        template = npc_template
    elif args.template == "math":
        template = math_template
    else:
        raise ValueError("Invalid template type. Choose from 'instruction', 'knowledge', 'npc', or 'math'.")

    # Load the dataset
    persona_dataset = load_dataset("proj-persona/PersonaHub", data_files="persona.jsonl")['train']
    if args.sample_size > 0:
        persona_dataset = persona_dataset[:args.sample_size]
    print(f"Total number of input personas: {len(persona_dataset['persona'])}")

    with open(args.output_path, "w") as out:
        for persona in tqdm(persona_dataset['persona']):
            persona = persona.strip()
            user_prompt = template.format(persona=persona)
            gpt4o_out_text = get_response(user_prompt)
            o = {"user_prompt": user_prompt, "input persona": persona, "synthesized text": gpt4o_out_text}
            out.write(json.dumps(o, ensure_ascii=False) + '\n')

    print(f"Outputted the results to: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize text using a specified model and template.")
    parser.add_argument('--sample_size', type=int, default=0, help='Number of samples to process from the dataset; Set it to 0 if you want to use the full set of 200k personas.')
    parser.add_argument(
        '--template', 
        type=str, 
        required=True, 
        choices=['instruction', 'knowledge', 'npc', 'math'], 
        help=(
            "Prompt templates. Choose from 'instruction', 'knowledge', 'math' or 'npc'. "
            "You can also add more customized templates in prompt_templates.py"
        )
    )
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file.')

    args = parser.parse_args()
    main(args)
