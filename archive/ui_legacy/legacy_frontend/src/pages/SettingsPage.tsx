import React, { useState } from 'react'
import { 
  Card, 
  Input, 
  Button, 
  Select,
  Switch,
  Alert,
  useTranslation
} from '@smolitux/core'

interface Settings {
  llmBackend: string
  apiKey: string
  maxTokens: number
  temperature: number
  enableLocalModels: boolean
  enableLogging: boolean
  language: string
}

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<Settings>({
    llmBackend: 'openai',
    apiKey: '',
    maxTokens: 1000,
    temperature: 0.7,
    enableLocalModels: false,
    enableLogging: true,
    language: 'de'
  })
  const [saved, setSaved] = useState(false)
  const t = useTranslation()

  const handleChange = (field: keyof Settings, value: any) => {
    setSettings({
      ...settings,
      [field]: value
    })
    setSaved(false)
  }

  const handleSave = () => {
    // In a real implementation, this would save to an API
    console.log('Saving settings:', settings)
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
  }

  const llmOptions = [
    { value: 'openai', label: 'OpenAI' },
    { value: 'lmstudio', label: 'LM Studio' },
    { value: 'local', label: 'Local Model' }
  ]

  const languageOptions = [
    { value: 'de', label: 'Deutsch' },
    { value: 'en', label: 'English' },
    { value: 'fr', label: 'Français' }
  ]

  return (
    <div className="settings-page">
      <h1>{t('settings.title')}</h1>
      
      {saved && (
        <Alert type="success" className="mb-2">
          {t('settings.saved')}
        </Alert>
      )}
      
      <Card className="settings-container">
        <h2>{t('settings.llmSettings')}</h2>
        
        <div className="settings-group">
          <label>{t('settings.llmBackend')}</label>
          <Select 
            options={llmOptions} 
            value={settings.llmBackend}
            onChange={(value) => handleChange('llmBackend', value)}
          />
        </div>
        
        <div className="settings-group">
          <label>{t('settings.apiKey')}</label>
          <Input 
            type="password"
            value={settings.apiKey}
            onChange={(e) => handleChange('apiKey', e.target.value)}
            placeholder={t('settings.apiKeyPlaceholder')}
          />
        </div>
        
        <div className="settings-group">
          <label>{t('settings.maxTokens')}</label>
          <Input 
            type="number"
            value={settings.maxTokens.toString()}
            onChange={(e) => handleChange('maxTokens', parseInt(e.target.value))}
            min="100"
            max="4000"
          />
        </div>
        
        <div className="settings-group">
          <label>{t('settings.temperature')}</label>
          <Input 
            type="range"
            value={settings.temperature.toString()}
            onChange={(e) => handleChange('temperature', parseFloat(e.target.value))}
            min="0"
            max="1"
            step="0.1"
          />
          <span>{settings.temperature}</span>
        </div>
        
        <div className="settings-group">
          <label>{t('settings.enableLocalModels')}</label>
          <Switch 
            checked={settings.enableLocalModels}
            onChange={() => handleChange('enableLocalModels', !settings.enableLocalModels)}
          />
        </div>
      </Card>
      
      <Card className="settings-container">
        <h2>{t('settings.systemSettings')}</h2>
        
        <div className="settings-group">
          <label>{t('settings.enableLogging')}</label>
          <Switch 
            checked={settings.enableLogging}
            onChange={() => handleChange('enableLogging', !settings.enableLogging)}
          />
        </div>
        
        <div className="settings-group">
          <label>{t('settings.language')}</label>
          <Select 
            options={languageOptions} 
            value={settings.language}
            onChange={(value) => handleChange('language', value)}
          />
        </div>
      </Card>
      
      <div className="settings-actions">
        <Button 
          variant="primary" 
          onClick={handleSave}
        >
          {t('settings.save')}
        </Button>
        <Button 
          variant="secondary"
          onClick={() => window.location.reload()}
        >
          {t('settings.cancel')}
        </Button>
      </div>
    </div>
  )
}

export default SettingsPage