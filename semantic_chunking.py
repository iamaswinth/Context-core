from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

llm=ChatOpenAI(model="gpt-4o", temperature=0)

tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""


prompt=f"""
you are a text chunking expert. Split this text into logical chunks.

Rules:
    - Each chunks should be around 200 characters or less
    - Split at natural topic boundaries
    - Keep related information together
    - Put "<<SPLIT>>"

    Text:
    {tesla_text}


"""

print("Asking ai to chunk the")

response=llm.invoke(prompt)
marked_text=response.content

chunks=marked_text.split("<<SPLIT>>")

clean_chunks=[]
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        clean_chunks.append(cleaned)

# Show results
print("\n🎯 AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()