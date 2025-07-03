"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const axios_1 = require("axios");
class ListAgents {
    constructor(endpoint, path = '/agents', headers = {}, timeout = 10000) {
        this.endpoint = endpoint;
        this.path = path;
        this.headers = headers;
        this.timeout = timeout;
    }
    run() {
        return __awaiter(this, void 0, void 0, function* () {
            var _a;
            const url = `${this.endpoint.replace(/\/$/, '')}${this.path}`;
            const opts = {
                url,
                method: 'GET',
                headers: this.headers,
                timeout: this.timeout,
            };
            try {
                const response = yield axios_1.default.request(opts);
                return response.data;
            }
            catch (err) {
                return { error: (_a = err.message) !== null && _a !== void 0 ? _a : String(err) };
            }
        });
    }
}
exports.default = ListAgents;
