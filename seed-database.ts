// Import Google's Gemini chat model and embeddings for AI text generation and vector creation
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
// Import structured output parser to ensure AI returns data in specific format
import { StructuredOutputParser } from "@langchain/core/output_parsers"
// Import MongoDB client for database connection
import { MongoClient } from "mongodb"
// Import MongoDB Atlas vector search for storing and searching embeddings
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
// Import Zod for data schema validation and type safety
import { z } from "zod"
// Load environment variables from .env file (API keys, connection strings)
import "dotenv/config"

// Create MongoDB client instance using connection string from environment variables
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string)

// Initialize Google Gemini chat model for generating synthetic furniture data
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",  // Use Gemini 1.5 Flash model
  temperature: 0.7,               // Set creativity level (0.7 = moderately creative)
  apiKey: process.env.GOOGLE_API_KEY, // Google API key from environment variables
})

// Define schema for furniture item structure using Zod validation
const itemSchema = z.object({
  item_id: z.string(),                    // Unique identifier for the item
  item_name: z.string(),                  // Name of the furniture item
  item_description: z.string(),           // Detailed description of the item
  brand: z.string(),                      // Brand/manufacturer name
  manufacturer_address: z.object({        // Nested object for manufacturer location
    street: z.string(),                   // Street address
    city: z.string(),                     // City name
    state: z.string(),                    // State/province
    postal_code: z.string(),              // ZIP/postal code
    country: z.string(),                  // Country name
  }),
  prices: z.object({                      // Nested object for pricing information
    full_price: z.number(),               // Regular price
    sale_price: z.number(),               // Discounted price
  }),
  categories: z.array(z.string()),        // Array of category tags
  user_reviews: z.array(                  // Array of customer reviews
    z.object({
      review_date: z.string(),            // Date of review
      rating: z.number(),                 // Numerical rating (1-5)
      comment: z.string(),                // Review text comment
    })
  ),
  notes: z.string(),                      // Additional notes about the item
})

// Create TypeScript type from Zod schema for type safety
type Item = z.infer<typeof itemSchema>

// Create parser that ensures AI output matches our item schema
const parser = StructuredOutputParser.fromZodSchema(z.array(itemSchema))

// Function to create database and collection before seeding
async function setupDatabaseAndCollection(): Promise<void> {
  console.log("Setting up database and collection...")

  // Get reference to the inventory_database database
  const db = client.db("inventory_database")

  // Create the items collection if it doesn't exist
  const collections = await db.listCollections({ name: "items" }).toArray()

  if (collections.length === 0) {
    await db.createCollection("items")
    console.log("Created 'items' collection in 'inventory_database' database")
  } else {
    console.log("'items' collection already exists in 'inventory_database' database")
  }
}

// Function to create vector search index
async function createVectorSearchIndex(): Promise<void> {
  try {
    const db = client.db("inventory_database")
    const collection = db.collection("items")
    await collection.dropIndexes()
    const vectorSearchIdx = {
      name: "vector_index",
      type: "vectorSearch",
      definition: {
        "fields": [
          {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 768,
            "similarity": "cosine"
          }
        ]
      }
    }
    console.log("Creating vector search index...")
    await collection.createSearchIndex(vectorSearchIdx);

    console.log("Successfully created vector search index");
  } catch (e) {
    console.error('Failed to create vector search index:', e);
  }
}

async function generateSyntheticData(): Promise<Item[]> {
  // Create detailed prompt instructing AI to generate furniture store data
  const prompt = `You are the official AI Business Assistant of **Chatsea**, a company that provides professional Facebook Page Automation and AI solutions.
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
Assistant: “জী! আমাদের অফিসিয়াল ওয়েবসাইট 👉**https://chatsea.is-great.net/**”.

  ${parser.getFormatInstructions()}`  // Add format instructions from parser

  // Log progress to console
  console.log("Generating synthetic data...")

  // Send prompt to AI and get response
  const response = await llm.invoke(prompt)
  // Parse AI response into structured array of Item objects
  return parser.parse(response.content as string)
}

// Function to create a searchable text summary from furniture item data
async function createItemSummary(item: Item): Promise<string> {
  // Return Promise for async compatibility (though this function is synchronous)
  return new Promise((resolve) => {
    // Extract manufacturer country information
    const manufacturerDetails = `Made in ${item.manufacturer_address.country}`
    // Join all categories into comma-separated string
    const categories = item.categories.join(", ")
    // Convert user reviews array into readable text format
    const userReviews = item.user_reviews
      .map(
        (review) =>
          `Rated ${review.rating} on ${review.review_date}: ${review.comment}`
      )
      .join(" ")  // Join multiple reviews with spaces
    // Create basic item information string
    const basicInfo = `${item.item_name} ${item.item_description} from the brand ${item.brand}`
    // Format pricing information
    const price = `At full price it costs: ${item.prices.full_price} USD, On sale it costs: ${item.prices.sale_price} USD`
    // Get additional notes
    const notes = item.notes

    // Combine all information into comprehensive summary for vector search
    const summary = `${basicInfo}. Manufacturer: ${manufacturerDetails}. Categories: ${categories}. Reviews: ${userReviews}. Price: ${price}. Notes: ${notes}`

    // Resolve promise with complete summary
    resolve(summary)
  })
}

// Main function to populate database with AI-generated furniture data
async function seedDatabase(): Promise<void> {
  try {
    // Establish connection to MongoDB Atlas
    await client.connect()
    // Ping database to verify connection works
    await client.db("admin").command({ ping: 1 })
    // Log successful connection
    console.log("You successfully connected to MongoDB!")

    // Setup database and collection
    await setupDatabaseAndCollection()

    // Create vector search index
    await createVectorSearchIndex()

    // Get reference to specific database
    const db = client.db("inventory_database")
    // Get reference to items collection
    const collection = db.collection("items")

    // Clear existing data from collection (fresh start)
    await collection.deleteMany({})
    console.log("Cleared existing data from items collection")

    // Generate new synthetic furniture data using AI
    const syntheticData = await generateSyntheticData()

    // Process each item: create summary and prepare for vector storage
    const recordsWithSummaries = await Promise.all(
      syntheticData.map(async (record) => ({
        pageContent: await createItemSummary(record),  // Create searchable summary
        metadata: { ...record },                         // Preserve original item data
      }))
    )

    // Store each record with vector embeddings in MongoDB
    for (const record of recordsWithSummaries) {
      // Create vector embeddings and store in MongoDB Atlas using Gemini
      await MongoDBAtlasVectorSearch.fromDocuments(
        [record],                    // Array containing single record
        new GoogleGenerativeAIEmbeddings({            // Google embedding model
          apiKey: process.env.GOOGLE_API_KEY,         // Google API key
          modelName: "text-embedding-004",            // Google's standard embedding model (768 dimensions)
        }),
        {
          collection,                // MongoDB collection reference
          indexName: "vector_index", // Name of vector search index
          textKey: "embedding_text", // Field name for searchable text
          embeddingKey: "embedding", // Field name for vector embeddings
        }
      )

      // Log progress for each successfully processed item
      console.log("Successfully processed & saved record:", record.metadata.item_id)
    }

    // Log completion of entire seeding process
    console.log("Database seeding completed")

  } catch (error) {
    // Log any errors that occur during database seeding
    console.error("Error seeding database:", error)
  } finally {
    // Always close database connection when finished (cleanup)
    await client.close()
  }
}

// Execute the database seeding function and handle any errors
seedDatabase().catch(console.error)
