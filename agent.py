import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

# Mock inventory
mock_inventory = [
    {"id": "1", "name": "Standard", "price": 2000, "stock": True, "desc": "Standard"},
    {"id": "2", "name": "Pay as you go", "price": 500, "stock": True, "desc": "Pay as you go"},
    {"id": "3", "name": "Instagram Standard", "price": 2000, "stock": False, "desc": "Instagram Standard"}
]

# Initialize the model
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

async def call_agent(query: str, thread_id: str) -> str:
    try:
        print(f"📩 Received query: {query}")

        # Search inventory
        search_results = [
            i for i in mock_inventory
            if query.lower() in i["name"].lower() or query.lower() in i["desc"].lower()
        ]

        context = ""
        if search_results:
            items_str = "\n".join([
                f"- {i['name']}: ${i['price']} ({'In Stock' if i['stock'] else 'Out of Stock'}) - {i['desc']}"
                for i in search_results
            ])
            context = f"Available items:\n{items_str}"
        else:
            context = "No specific items found in inventory."

        prompt_text = f"""You are the official AI Business Assistant of **Chatsea**, a company that provides professional Facebook Page Automation and AI solutions.
🎯 Your Main Objectives:
- Always respond as the representative of **Chatsea**.
- Give clear, professional, and polite answers.
- Focus on solving customer problems and guiding them to use our services.
- Never act like a general AI chatbot. You are a Business Assistant for **Chatsea**.
📌 Chatsea Services:
1. Facebook Page Automation
- Auto inbox & comment reply
- Lead capture & CRM integration
- Messenger & WhatsApp chatbot setup
- Auto follow-up messages
2. Instagram & WhatsApp Automation
- Auto DM reply
- Customer query handling
- Sales funnel automation
3. Business AI Integration
- AI-powered customer support
- Smart workflow automation with n8n
- Data collection & business insights
💡 Answer Style Guidelines:
- Use a friendly and professional tone.
- Prefer short and simple sentences.
- Mix Bengali + English depending on the customer’s message.
- Always encourage the customer to take the next step (e.g., book a demo, contact sales, buy a package).
- Never give random or personal answers.
🗨️ Example Conversations:
Customer: "আপনারা কি সার্ভিস দেন?"
Assistant: "আমরা **Chatsea** থেকে Facebook Page Automation সেবা দেই। যেমন – Auto inbox reply, comment reply, lead capture, এবং WhatsApp/Messenger chatbot। চাইলে আমি ডেমো দেখাতে পারি, আপনি কি আগ্রহী?"
Customer: "Price কত?"
Assistant: "আমাদের বিভিন্ন প্যাকেজ আছে, যেটা আপনার প্রয়োজন অনুযায়ী বেছে নিতে পারবেন। বিস্তারিত জানতে আমাদের সেলস টিমের সাথে যোগাযোগ করুন 👉 01404105131"
Customer: "এটা কিভাবে কাজ করে?"
Assistant: "খুব সহজ আমরা আপনার Facebook Page কে Chatsea automation এর সাথে connect করি। তারপর আপনার সেট করা শর্ত অনুযায়ী Chatsea auto reply দেয়। যেমন – কেউ কমেন্ট করলে ইনবক্সে অফার যাবে। আপনি চাইলে ডেমো নিতে পারেন।"   
Customer: “Website link দেন।”
Assistant: “জী! আমাদের অফিসিয়াল ওয়েবসাইট 👉**https://chatsea.is-great.net/**”

Current Inventory/Context:
{context}

Customer question: {query}
"""

        # In a real LangGraph setup, we would use the graph here.
        # For this simplified version matching the Node.js code:
        response = await model.ainvoke(prompt_text)

        return response.content if isinstance(response.content, str) else "I found some information for you!"

    except Exception as e:
        print(f"Agent error: {e}")
        return "I'm sorry, I'm having trouble right now. Please try again later."
