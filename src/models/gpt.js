import OpenAIApi from 'openai';
import { getKey, hasKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';

export class GPT {
    constructor(modelName, url = null) {
        this.modelName = modelName;
        this.baseUrl = url;
        this.apiKey = null; // Will store the appropriate API key


        let config = {};

        if (this.modelName && this.modelName.startsWith("grok")) {
            config.baseURL = "https://api.x.ai/v1"; // Grok's base URL
            this.apiKey = getKey('XAI_API_KEY');       // Use xAI key for Grok
        } else {  //For other OpenAI models
            if (this.baseUrl) {
                config.baseURL = this.baseUrl; 
            }
            if (hasKey('OPENAI_ORG_ID')) {
                config.organization = getKey('OPENAI_ORG_ID');
            }
            this.apiKey = getKey('OPENAI_API_KEY'); // Use OpenAI Key for other models.
        }


        config.apiKey = this.apiKey;  // Set the apiKey in the config


        this.openai = new OpenAIApi(config);
    }

    sync sendRequest(turns, systemMessage, stop_seq = '***', retryCount = 0) {
        // Retry logic to handle rate limits or transient errors
        if (retryCount > 5) {
            console.error('Maximum retry attempts reached for OpenAI API.');
            return 'Error: Too many retry attempts.';
        }

        let messages = [{ role: 'system', content: systemMessage }].concat(turns);

        const pack = {
            model: this.modelName || "gpt-3.5-turbo",
            messages,
            stop: stop_seq,
        };


        if (this.modelName && this.modelName.includes('o1')) { // Handle GPT-3.5-turbo-o1 format
            pack.messages = strictFormat(messages);
            delete pack.stop; // o1 models don't use stop sequences
        }

        let res = null;
        try {
            console.log(`Awaiting OpenAI API response... (Model: ${this.modelName}, Retry: ${retryCount})`);

            let completion = await this.openai.chat.completions.create(pack);
            if (completion.choices[0].finish_reason === 'length') {
                console.warn('Context length exceeded. Trying with shorter context...');
                // Remove the oldest turn and retry
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq, retryCount + 1);
            }

            console.log('Received.');
            res = completion.choices[0].message.content;

        } catch (err) {
            if (err.message === 'Context length exceeded' || err.code === 'context_length_exceeded') {
                console.warn('Context length exceeded. Trying with shorter context...');
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq, retryCount + 1);
            } else if (
                err.response && (err.response.status === 429 || err.response.status >= 500)
            ) {
                console.warn(`OpenAI API rate limit or server error (Status ${err.response.status}). Retrying...`);
                const retryDelay = (2 ** retryCount) * 1000; // Exponential backoff
                await new Promise(resolve => setTimeout(resolve, retryDelay));
                return await this.sendRequest(turns, systemMessage, stop_seq, retryCount + 1);

            }
            else {
                console.error('OpenAI API Error:', err);  // Log full error for debugging
                res = 'My brain disconnected, try again.';
            }
        }
        return res;
    }

    async embed(text) {
      if (this.modelName && this.modelName.startsWith("grok")) {
        console.log("There is no current support for embeddings with Grok. Text provided:", text);
        throw new Error('Embeddings are not supported by Grok');
      }
      try {
          const embedding = await this.openai.embeddings.create({
              model: "text-embedding-ada-002", // Most capable embedding model
              input: text,
          });
          return embedding.data[0].embedding;
      } catch (error) {
          console.error("Error creating embedding:", error);
          throw new Error("Embedding creation failed.");
      }
  }
}