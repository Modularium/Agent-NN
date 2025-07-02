import axios from 'axios';

export default class AgentNN {
  constructor(
    private endpoint: string,
    private taskType = 'chat',
    private headers: Record<string, string> = {},
    private timeout = 10000,
    private method: 'POST' | 'GET' | 'PUT' | 'DELETE' = 'POST',
    private path = '/task'
  ) {}

  async run(payload: unknown): Promise<any> {
    const url = `${this.endpoint.replace(/\/$/, '')}${this.path}`;
    const body = { task_type: this.taskType, input: payload };
    const response = await axios.request({
      url,
      method: this.method,
      data: body,
      headers: this.headers,
      timeout: this.timeout,
    });
    const data = response.data;
    return data.result ?? data;
  }
}
