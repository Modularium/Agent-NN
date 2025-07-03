export default class CreateTask {
    private endpoint;
    private taskType;
    private input;
    private path;
    private method;
    private headers;
    private timeout;
    constructor(endpoint: string, taskType?: string, input?: unknown, path?: string, method?: 'POST' | 'GET' | 'PUT' | 'DELETE', headers?: Record<string, string>, timeout?: number);
    run(): Promise<any>;
}
