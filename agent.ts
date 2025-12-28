// agent.ts - Simplified version
import "dotenv/config"
import { ChatGroq } from "@langchain/groq"

// Mock inventory
const mockInventory = [
  { id: "1", name: "Standard", price: 2000, stock: true, desc: "Standard" },
  { id: "2", name: "Pay as you go", price: 500, stock: true, desc: "Pay as you go" },
  { id: "3", name: "Instagram Standard", price: 2000, stock: false, desc: "Instagram Standard" }
]

// Initialize the model
const model = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0,
  apiKey: process.env.GROQ_API_KEY
})

// Main function to call the agent
export async function callAgent(query: string, threadId: string): Promise<string> {
  try {
    console.log(`📩 Received query: ${query}`);

    // Search inventory
    const searchResults = mockInventory
      .filter(i =>
        i.name.toLowerCase().includes(query.toLowerCase()) ||
        i.desc.toLowerCase().includes(query.toLowerCase())
      )

    let context = "";
    if (searchResults.length > 0) {
      context = `Available items:\n${searchResults.map(i =>
        `- ${i.name}: $${i.price} (${i.stock ? 'In Stock' : 'Out of Stock'}) - ${i.desc}`
      ).join('\n')}`;
    }

    const prompt = `You are the official AI Business Assistant of **Chatsea**, a company that provides professional Facebook Page Automation and AI solutions.
${context ? context : 'No specific items found in inventory.'}

Customer question: ${query}

Provide a helpful response about the ChatSea.`;

    const response = await model.invoke(prompt);

    return typeof response.content === "string"
      ? response.content
      : "I found some information for you!";

  } catch (error: any) {
    console.error("Agent error:", error.message);
    return "I'm sorry, I'm having trouble right now. Please try again later.";
  }
}