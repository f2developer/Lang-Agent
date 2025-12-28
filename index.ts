// index.ts
import "dotenv/config"
import express from "express"
import cors from "cors"
import { callAgent } from "./agent"
import { randomUUID } from "crypto"

const app = express()
app.use(cors())
app.use(express.json())

app.get("/", (req, res) => res.send("Chat Agent Live!"))

app.post("/chat", async (req, res) => {
  const { message, threadId: existingThreadId } = req.body
  if (!message) return res.status(400).json({ error: "message required" })

  const threadId = existingThreadId || randomUUID()
  try {
    const reply = await callAgent(message, threadId)
    res.json({ threadId, reply })
  } catch (e: any) {
    res.status(500).json({ error: e.message })
  }
})

app.listen(8000, () => console.log("http://localhost:8000"))