export default class ListAgents {
    private endpoint;
    private path;
    private headers;
    private timeout;
    constructor(endpoint: string, path?: string, headers?: Record<string, string>, timeout?: number);
    run(): Promise<any>;
}
