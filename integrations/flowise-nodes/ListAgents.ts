import axios, { AxiosRequestConfig } from 'axios';

export default class ListAgents {
  constructor(
    private endpoint: string,
    private path = '/agents',
    private headers: Record<string, string> = {},
    private timeout = 10000
  ) {}

  async run(): Promise<any> {
    const url = `${this.endpoint.replace(/\/$/, '')}${this.path}`;
    const opts: AxiosRequestConfig = {
      url,
      method: 'GET',
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
