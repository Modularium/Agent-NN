import axios, { AxiosRequestConfig } from 'axios';

export default class AgentNN {
  constructor(
    private endpoint: string,
    private taskType = 'chat',
    private headers: Record<string, string> = {},
    private timeout = 10000,
    private method: 'POST' | 'GET' | 'PUT' | 'DELETE' = 'POST',
    private path = '/task',
    private auth?: { username?: string; password?: string; token?: string }
  ) {}

  async run(payload: unknown): Promise<any> {
    const url = `${this.endpoint.replace(/\/$/, '')}${this.path}`;
    const body = { task_type: this.taskType, input: payload };
    const headers = { ...this.headers } as Record<string, string>;
    if (this.auth?.token) {
      headers['Authorization'] = `Bearer ${this.auth.token}`;
    }
    const opts: AxiosRequestConfig = {
      url,
      method: this.method,
      data: body,
      headers: this.headers,
      timeout: this.timeout,
    };
    if (this.auth?.username && this.auth?.password) {
      opts['auth'] = {
        username: this.auth.username,
        password: this.auth.password,
      };
    }
    try {
      const response = await axios.request(opts);
      const data = response.data;
      return data.result ?? data;
    } catch (err: any) {
      return { error: err.message ?? String(err) };
    }
  }
}
