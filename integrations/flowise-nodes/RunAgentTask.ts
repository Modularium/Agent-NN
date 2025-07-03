import axios, { AxiosRequestConfig } from 'axios';

export default class RunAgentTask {
  constructor(
    private endpoint: string,
    private description: string,
    private domain: string | null = null,
    private path = '/flowise/run_task',
    private method: 'POST' | 'GET' | 'PUT' | 'DELETE' = 'POST',
    private headers: Record<string, string> = {},
    private timeout = 10000
  ) {}

  async run(): Promise<any> {
    const url = `${this.endpoint.replace(/\/$/, '')}${this.path}`;
    const body = { description: this.description, domain: this.domain };
    const opts: AxiosRequestConfig = {
      url,
      method: this.method,
      data: body,
      headers: this.headers,
      timeout: this.timeout,
    };
    try {
      const response = await axios.request(opts);
      return response.data;
    } catch (err: any) {
      return { error: err.message ?? String(err) };
    }
  }
}
