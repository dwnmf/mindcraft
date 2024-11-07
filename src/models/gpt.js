import OpenAIApi from 'openai';
import { getKey, hasKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';

export class GPT {
    constructor(modelName, url = null) {
        this.modelName = modelName;
        this.baseUrl = url;
        this.apiKey = null;

        const config = {};

        if (this.modelName && this.modelName.startsWith("grok")) {
            config.baseURL = "https://api.x.ai/v1";
            this.apiKey = getKey('XAI_API_KEY');
        } else {
            if (this.baseUrl) {
                config.baseURL = this.baseUrl;
            }
            if (hasKey('OPENAI_ORG_ID')) {
                config.organization = getKey('OPENAI_ORG_ID');
            }
            this.apiKey = getKey('OPENAI_API_KEY');
        }

        config.apiKey = this.apiKey;
        this.openai = new OpenAIApi(config);
    }

    async sendRequest(turns, systemMessage, stop_seq = ['***'], retryCount = 0) {
        if (retryCount > 5) {
            console.error('Maximum retry attempts reached.');
            return 'Error: Too many retry attempts.';
        }

        const messages = [{ role: 'system', content: systemMessage }, ...turns];

        const pack = {
            model: this.modelName || "gpt-3.5-turbo-0613",
            messages,
            stop: stop_seq,
        };

        if (this.modelName && this.modelName.includes('o1')) {
            pack.messages = strictFormat(messages);
            delete pack.stop;
        }

        try {
            console.log(`Awaiting API response... (Model: ${this.modelName}, Retry: ${retryCount})`);
            const completion = await this.openai.createChatCompletion(pack);
            return completion.data.choices[0].message.content; // Directly return the content
        } catch (err) {
            if (err.message === 'Context length exceeded' || err.code === 'context_length_exceeded') {
                console.warn('Context length exceeded. Trying with shorter context...');
                return await this.sendRequest(turns.slice(1), systemMessage, stop_seq, retryCount + 1);
            } else if (err.response && (err.response.status === 429 || err.response.status >= 500)) {
                console.warn(`API rate limit or server error (Status ${err.response.status}). Retrying...`);
                const retryDelay = (2 ** retryCount) * 1000;
                await new Promise(resolve => setTimeout(resolve, retryDelay));
                return await this.sendRequest(turns, systemMessage, stop_seq, retryCount + 1);
            } else {
                console.error('API Error:', err);
                return 'My brain disconnected, try again.';
            }
        }
    }


    async embed(text) {
        if (this.modelName && this.modelName.startsWith("grok")) {
            console.log("Embeddings are not yet supported for Grok. Text provided:", text);
            throw new Error('Embeddings are not supported by Grok');
        }

        try {
            const embedding = await this.openai.createEmbedding({
                model: "text-embedding-ada-002",
                input: text,
            });
            return embedding.data.data[0].embedding;
        } catch (error) {
            console.error("Error creating embedding:", error);
            throw new Error("Embedding creation failed.");
        }
    }
}