import axios from 'axios';

export default class AgentNN {
  constructor(private endpoint: string) {}

  async run(question: string): Promise<string> {
    const { data } = await axios.post(`${this.endpoint}/task`, {
      task_type: 'chat',
      input: question,
    });
    return data.result as string;
  }
}
