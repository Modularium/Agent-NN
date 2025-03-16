<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <!-- Background -->
  <rect width="800" height="600" fill="#f5f7fa" rx="10" ry="10"/>
  
  <!-- Title -->
  <text x="400" y="40" font-family="Arial" font-size="24" text-anchor="middle" font-weight="bold" fill="#333">Smolit's Agent-Framework Systemarchitektur</text>
  
  <!-- User -->
  <circle cx="400" cy="100" r="30" fill="#4299e1"/>
  <text x="400" y="105" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Benutzer</text>
  
  <!-- Web Interface -->
  <rect x="300" y="150" width="200" height="50" rx="10" ry="10" fill="#63b3ed"/>
  <text x="400" y="180" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Chatbot-Interface</text>
  
  <!-- Supervisor Agent -->
  <rect x="300" y="250" width="200" height="50" rx="10" ry="10" fill="#4c51bf"/>
  <text x="400" y="280" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Supervisor-Agent</text>
  
  <!-- Domain Agents -->
  <rect x="100" y="350" width="120" height="50" rx="10" ry="10" fill="#805ad5"/>
  <text x="160" y="380" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Finanz-Agent</text>
  
  <rect x="250" y="350" width="120" height="50" rx="10" ry="10" fill="#805ad5"/>
  <text x="310" y="380" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Tech-Agent</text>
  
  <rect x="400" y="350" width="120" height="50" rx="10" ry="10" fill="#805ad5"/>
  <text x="460" y="380" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Marketing-Agent</text>
  
  <rect x="550" y="350" width="120" height="50" rx="10" ry="10" fill="#805ad5"/>
  <text x="610" y="380" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Web-Agent</text>
  
  <!-- Knowledge Bases -->
  <rect x="100" y="450" width="120" height="50" rx="10" ry="10" fill="#38b2ac"/>
  <text x="160" y="480" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Finanz-Wissen</text>
  
  <rect x="250" y="450" width="120" height="50" rx="10" ry="10" fill="#38b2ac"/>
  <text x="310" y="480" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Tech-Wissen</text>
  
  <rect x="400" y="450" width="120" height="50" rx="10" ry="10" fill="#38b2ac"/>
  <text x="460" y="480" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Marketing-Wissen</text>
  
  <rect x="550" y="450" width="120" height="50" rx="10" ry="10" fill="#38b2ac"/>
  <text x="610" y="480" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Web-Crawling</text>
  
  <!-- External Systems -->
  <rect x="50" y="550" width="700" height="30" rx="10" ry="10" fill="#2c5282"/>
  <text x="400" y="570" font-family="Arial" font-size="12" text-anchor="middle" fill="white">Externe APIs und Datenquellen</text>
  
  <!-- Connections -->
  <!-- User to Interface -->
  <line x1="400" y1="130" x2="400" y2="150" stroke="#718096" stroke-width="2"/>
  <!-- Interface to Supervisor -->
  <line x1="400" y1="200" x2="400" y2="250" stroke="#718096" stroke-width="2"/>
  <!-- Supervisor to Agents -->
  <line x1="400" y1="300" x2="160" y2="350" stroke="#718096" stroke-width="2"/>
  <line x1="400" y1="300" x2="310" y2="350" stroke="#718096" stroke-width="2"/>
  <line x1="400" y1="300" x2="460" y2="350" stroke="#718096" stroke-width="2"/>
  <line x1="400" y1="300" x2="610" y2="350" stroke="#718096" stroke-width="2"/>
  <!-- Agents to Knowledge Bases -->
  <line x1="160" y1="400" x2="160" y2="450" stroke="#718096" stroke-width="2"/>
  <line x1="310" y1="400" x2="310" y2="450" stroke="#718096" stroke-width="2"/>
  <line x1="460" y1="400" x2="460" y2="450" stroke="#718096" stroke-width="2"/>
  <line x1="610" y1="400" x2="610" y2="450" stroke="#718096" stroke-width="2"/>
  <!-- Knowledge Bases to External Systems -->
  <line x1="160" y1="500" x2="160" y2="550" stroke="#718096" stroke-width="2"/>
  <line x1="310" y1="500" x2="310" y2="550" stroke="#718096" stroke-width="2"/>
  <line x1="460" y1="500" x2="460" y2="550" stroke="#718096" stroke-width="2"/>
  <line x1="610" y1="500" x2="610" y2="550" stroke="#718096" stroke-width="2"/>
  
  <!-- Legend -->
  <rect x="650" y="100" width="130" height="140" rx="5" ry="5" fill="white" stroke="#e2e8f0" stroke-width="1"/>
  <text x="715" y="120" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold" fill="#333">Legende</text>
  
  <circle cx="670" cy="140" r="10" fill="#4299e1"/>
  <text x="720" y="145" font-family="Arial" font-size="12" fill="#333">Benutzer</text>
  
  <rect x="660" y="160" width="20" height="20" rx="5" ry="5" fill="#63b3ed"/>
  <text x="720" y="175" font-family="Arial" font-size="12" fill="#333">Interface</text>
  
  <rect x="660" y="190" width="20" height="20" rx="5" ry="5" fill="#805ad5"/>
  <text x="720" y="205" font-family="Arial" font-size="12" fill="#333">Agenten</text>
  
  <rect x="660" y="220" width="20" height="20" rx="5" ry="5" fill="#38b2ac"/>
  <text x="720" y="235" font-family="Arial" font-size="12" fill="#333">Wissensbasen</text>
</svg>
