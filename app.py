from transformers import AutoTokenizer, OPTForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
model = OPTForCausalLM.from_pretrained("facebook/galactica-6.7b", device_map="auto")

def generate(input_text):
  input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
  outputs = model.generate(input_ids, max_length=100)
  return tokenizer.decode(outputs[0])

input_text = """translate English to Spanish: Hello world"""
generate(input_text)