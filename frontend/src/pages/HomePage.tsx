import { Card, useTranslation, Button } from '@smolitux/core'
import { Link } from 'react-router-dom'

const HomePage = () => {
  const t = useTranslation()

  return (
    <div className="home-page">
      <div className="hero-section">
        <h1>{t('home.title')}</h1>
        <p className="hero-description">{t('home.description')}</p>
        <div className="hero-actions">
          <Link to="/chat">
            <Button variant="primary" size="lg">
              {t('home.startChat')}
            </Button>
          </Link>
          <Link to="/agents">
            <Button variant="secondary" size="lg">
              {t('home.exploreAgents')}
            </Button>
          </Link>
        </div>
      </div>

      <div className="features-section">
        <h2 className="section-title">{t('home.features')}</h2>
        <div className="features-grid">
          <Card className="feature-card">
            <h3>{t('home.features.multiAgent.title')}</h3>
            <p>{t('home.features.multiAgent.description')}</p>
          </Card>
          <Card className="feature-card">
            <h3>{t('home.features.neuralNetwork.title')}</h3>
            <p>{t('home.features.neuralNetwork.description')}</p>
          </Card>
          <Card className="feature-card">
            <h3>{t('home.features.knowledgeManagement.title')}</h3>
            <p>{t('home.features.knowledgeManagement.description')}</p>
          </Card>
          <Card className="feature-card">
            <h3>{t('home.features.llmIntegration.title')}</h3>
            <p>{t('home.features.llmIntegration.description')}</p>
          </Card>
        </div>
      </div>

      <div className="getting-started-section">
        <h2 className="section-title">{t('home.gettingStarted')}</h2>
        <Card className="getting-started-card">
          <ol className="steps-list">
            <li>{t('home.steps.chat')}</li>
            <li>{t('home.steps.task')}</li>
            <li>{t('home.steps.agents')}</li>
            <li>{t('home.steps.results')}</li>
          </ol>
          <Link to="/chat">
            <Button variant="primary">
              {t('home.startNow')}
            </Button>
          </Link>
        </Card>
      </div>
    </div>
  )
}

export default HomePage