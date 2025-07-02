import axios from 'axios';

export default class AgentNN {
  constructor(
    private endpoint: string,
    private taskType = 'chat',
    private headers: Record<string, string> = {},
    private timeout = 10000
  ) {}

  async run(payload: unknown): Promise<any> {
    const { data } = await axios.post(
      `${this.endpoint}/task`,
      {
        task_type: this.taskType,
        input: payload,
      },
      {
        headers: this.headers,
        timeout: this.timeout,
      }
    );
    return data.result ?? data;
  }
}
