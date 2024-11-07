import { OpenAIApi, Configuration } from 'openai';
import { getKey, hasKey } from '../utils/keys.js';
import { strictFormat } from '../utils/text.js';

export class GPT {
    constructor(modelName, url = null) {
        this.modelName = modelName;
        this.baseUrl = url;
        this.apiKey = null;

        const config = {};

        if (this.modelName && this.modelName.startsWith("grok")) {
            config.basePath = "https://api.x.ai/v1";
            this.apiKey = getKey('XAI_API_KEY');
        } else {
            if (hasKey('OPENAI_ORG_ID')) {
                config.organization = getKey('OPENAI_ORG_ID');
            }
            this.apiKey = getKey('OPENAI_API_KEY');
        }

        const openAIConfig = new Configuration({
            apiKey: this.apiKey,
            organization: config.organization,
            basePath: config.basePath,
        });

        this.openai = new OpenAIApi(openAIConfig);
    }

    async sendRequest(turns, systemMessage, stop_seq = '***', retryCount = 0) {
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
            const response = completion.data.choices[0].message.content;
            return response;
        } catch (err) {
            console.error('API Error:', err);
            return 'Request failed, please try again later.';
        }
    }

    async embed(text) {
        if (this.modelName && this.modelName.startsWith("grok")) {
            throw new Error('Embeddings not supported for Grok.');
        }

        try {
            const embedding = await this.openai.createEmbedding({
                model: "text-embedding-ada-002",
                input: text,
            });
            return embedding.data.data[0].embedding;
        } catch (error) {
            console.error("Embedding error:", error);
            throw new Error("Embedding creation failed.");
        }
    }
}
