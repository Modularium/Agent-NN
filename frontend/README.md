# Agent-NN Frontend

This is the frontend application for the Agent-NN system, built with React, TypeScript, and Smolitux-UI.

## Features

- Chat interface for interacting with the Agent-NN system
- Agent management and visualization
- Task history and details
- System settings and configuration

## Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install
# or
yarn install
```

### Development

```bash
# Start development server
npm run dev
# or
yarn dev
```

The application will be available at http://localhost:3000.

### Building for Production

```bash
# Build for production
npm run build
# or
yarn build
```

The build artifacts will be stored in the `dist/` directory.

## Project Structure

```
frontend/
├── public/              # Static files
├── src/
│   ├── components/      # Reusable components
│   ├── layouts/         # Layout components
│   ├── pages/           # Page components
│   ├── styles/          # CSS styles
│   ├── types/           # TypeScript type definitions
│   ├── App.tsx          # Main application component
│   └── main.tsx         # Entry point
├── index.html           # HTML template
├── package.json         # Dependencies and scripts
├── tsconfig.json        # TypeScript configuration
└── vite.config.ts       # Vite configuration
```

## Technologies

- [React](https://reactjs.org/)
- [TypeScript](https://www.typescriptlang.org/)
- [Vite](https://vitejs.dev/)
- [Smolitux-UI](https://github.com/EcoSphereNetwork/smolitux-ui)
- [React Router](https://reactrouter.com/)