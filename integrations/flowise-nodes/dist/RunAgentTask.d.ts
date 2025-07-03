export default class RunAgentTask {
    private endpoint;
    private description;
    private domain;
    private path;
    private method;
    private headers;
    private timeout;
    constructor(endpoint: string, description: string, domain?: string | null, path?: string, method?: 'POST' | 'GET' | 'PUT' | 'DELETE', headers?: Record<string, string>, timeout?: number);
    run(): Promise<any>;
}
