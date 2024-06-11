import csv
import time
from datetime import datetime
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def generate_dialogue(model, tokenizer, prompt, max_length=75):
    
    ## hyper tuning the model  ##
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  
        do_sample=True,  
        top_k=50,  
        top_p=0.95  
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main(csv_path, products, dilgoue = 10):

    ##pre-trained transfomer (gpt-2) from hugging faces
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    
    dialogues = []
    num_dialogues = dilgoue  # Generating n dilgoues

    start_time = time.time()

    for i in range(num_dialogues):
        for product in products:

            # inital prompts between sale's man and customer
            user_prompt = f"User: Hi! I'm looking for more information about your {product}. Can you tell me about it?\nSalesman:"
            salesman_prompt = f"Salesman: Hello! How can I help you today? Are you interested in our {product}?\nUser:"

            # Generate dialogue 
            salesman_response = generate_dialogue(model, tokenizer, salesman_prompt)
            user_response = generate_dialogue(model, tokenizer, user_prompt)

            # Ensure responses are within the 50-75 word limit
            if 50 <= len(salesman_response.split()) <= 75 and 50 <= len(user_response.split()) <= 75:
                
                # Create timestamp for each conversation
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Append to dialogues to a list
                dialogues.append([salesman_response, user_response, timestamp])

        dialogues.append(['*' * 10])

    end_time = time.time()
    compute_time = end_time - start_time

    # Save to CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Salesman", "User", "TimeStamp"])
        writer.writerows(dialogues)

    print(f"Data generation complete in {compute_time:.2f} seconds.")

if __name__ == "__main__":

    csv_path = 'conversation.csv' 

    ## define the products
    products = ["Smartphone iphone 6s", "Home Security System", "Eco-friendly Detergent"]
    dilgoue = 10
    main(csv_path, products, dilgoue)
