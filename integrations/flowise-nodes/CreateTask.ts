import axios, { AxiosRequestConfig } from 'axios';

export default class CreateTask {
  constructor(
    private endpoint: string,
    private taskType = 'chat',
    private input: unknown = {},
    private path = '/tasks',
    private method: 'POST' | 'GET' | 'PUT' | 'DELETE' = 'POST',
    private headers: Record<string, string> = {},
    private timeout = 10000
  ) {}

  async run(): Promise<any> {
    const url = `${this.endpoint.replace(/\/$/, '')}${this.path}`;
    const body = { task_type: this.taskType, input: this.input };
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
