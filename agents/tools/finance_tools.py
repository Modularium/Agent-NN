from typing import Dict, Any, Optional, List, Union
from langchain.tools.base import BaseTool
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class StockDataTool(BaseTool):
    """Tool for fetching stock market data."""
    
    name = "stock_data"
    description = "Get historical stock data and basic analysis"
    
    def _run(self, query: str) -> str:
        """Run stock data query.
        
        Args:
            query: Stock symbol or analysis request
            
        Returns:
            str: Query results
        """
        try:
            # Parse query
            parts = query.split()
            symbol = parts[0].upper()
            
            # Get stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo")
            
            if not len(hist):
                return f"No data found for symbol: {symbol}"
                
            # Calculate basic metrics
            current_price = hist['Close'][-1]
            price_change = (current_price - hist['Close'][0]) / hist['Close'][0] * 100
            volume = hist['Volume'].mean()
            
            # Calculate technical indicators
            sma_20 = hist['Close'].rolling(window=20).mean()
            rsi = self._calculate_rsi(hist['Close'])
            
            analysis = {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "price_change_30d": round(price_change, 2),
                "avg_volume": int(volume),
                "sma_20": round(sma_20[-1], 2),
                "rsi": round(rsi[-1], 2)
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error: {str(e)}"
            
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            periods: RSI periods
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class FinancialCalculatorTool(BaseTool):
    """Tool for financial calculations."""
    
    name = "financial_calculator"
    description = "Perform financial calculations (NPV, IRR, loan payments, etc.)"
    
    def _run(self, query: Dict[str, Any]) -> str:
        """Run financial calculation.
        
        Args:
            query: Calculation parameters
            
        Returns:
            str: Calculation results
        """
        try:
            calc_type = query.get("type", "").lower()
            
            if calc_type == "loan":
                return self._calculate_loan_payment(
                    principal=query.get("principal", 0),
                    rate=query.get("rate", 0),
                    years=query.get("years", 0)
                )
                
            elif calc_type == "npv":
                return self._calculate_npv(
                    rate=query.get("rate", 0),
                    cashflows=query.get("cashflows", [])
                )
                
            elif calc_type == "irr":
                return self._calculate_irr(
                    cashflows=query.get("cashflows", [])
                )
                
            else:
                return f"Unknown calculation type: {calc_type}"
                
        except Exception as e:
            return f"Calculation Error: {str(e)}"
            
    def _calculate_loan_payment(self,
                              principal: float,
                              rate: float,
                              years: int) -> str:
        """Calculate loan payment.
        
        Args:
            principal: Loan amount
            rate: Annual interest rate (%)
            years: Loan term in years
            
        Returns:
            str: Payment details
        """
        rate = rate / 100 / 12  # Monthly rate
        n_payments = years * 12  # Total payments
        
        # Calculate monthly payment
        payment = principal * (rate * (1 + rate)**n_payments) / ((1 + rate)**n_payments - 1)
        
        # Calculate total interest
        total_paid = payment * n_payments
        total_interest = total_paid - principal
        
        result = {
            "monthly_payment": round(payment, 2),
            "total_paid": round(total_paid, 2),
            "total_interest": round(total_interest, 2)
        }
        
        return json.dumps(result, indent=2)
        
    def _calculate_npv(self,
                      rate: float,
                      cashflows: List[float]) -> str:
        """Calculate Net Present Value.
        
        Args:
            rate: Discount rate (%)
            cashflows: List of cash flows
            
        Returns:
            str: NPV calculation
        """
        rate = rate / 100
        npv = 0
        
        for i, cf in enumerate(cashflows):
            npv += cf / (1 + rate)**i
            
        result = {
            "npv": round(npv, 2),
            "cashflows": len(cashflows),
            "rate": rate * 100
        }
        
        return json.dumps(result, indent=2)
        
    def _calculate_irr(self, cashflows: List[float]) -> str:
        """Calculate Internal Rate of Return.
        
        Args:
            cashflows: List of cash flows
            
        Returns:
            str: IRR calculation
        """
        try:
            # Use numpy's IRR function
            irr = np.irr(cashflows)
            
            result = {
                "irr": round(irr * 100, 2),
                "cashflows": len(cashflows)
            }
            
            return json.dumps(result, indent=2)
            
        except Exception:
            return "Could not calculate IRR"

class PortfolioAnalysisTool(BaseTool):
    """Tool for portfolio analysis."""
    
    name = "portfolio_analysis"
    description = "Analyze investment portfolio (risk, returns, allocation)"
    
    def _run(self, portfolio: Dict[str, Any]) -> str:
        """Run portfolio analysis.
        
        Args:
            portfolio: Portfolio details
            
        Returns:
            str: Analysis results
        """
        try:
            # Get portfolio data
            symbols = list(portfolio.get("holdings", {}).keys())
            weights = list(portfolio.get("holdings", {}).values())
            
            if not symbols:
                return "Empty portfolio"
                
            # Get historical data
            data = yf.download(
                symbols,
                start=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                end=datetime.now().strftime('%Y-%m-%d')
            )['Adj Close']
            
            # Calculate returns
            returns = data.pct_change()
            
            # Calculate portfolio metrics
            portfolio_return = self._calculate_portfolio_return(returns, weights)
            portfolio_risk = self._calculate_portfolio_risk(returns, weights)
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_return, portfolio_risk)
            
            analysis = {
                "annual_return": round(portfolio_return * 100, 2),
                "annual_risk": round(portfolio_risk * 100, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "holdings": len(symbols)
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Analysis Error: {str(e)}"
            
    def _calculate_portfolio_return(self,
                                  returns: pd.DataFrame,
                                  weights: List[float]) -> float:
        """Calculate portfolio return.
        
        Args:
            returns: Asset returns
            weights: Asset weights
            
        Returns:
            float: Portfolio return
        """
        # Calculate average returns
        avg_returns = returns.mean()
        
        # Calculate portfolio return
        port_return = np.sum(avg_returns * weights) * 252
        return port_return
        
    def _calculate_portfolio_risk(self,
                                returns: pd.DataFrame,
                                weights: List[float]) -> float:
        """Calculate portfolio risk.
        
        Args:
            returns: Asset returns
            weights: Asset weights
            
        Returns:
            float: Portfolio risk
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252
        
        # Calculate portfolio variance
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        return np.sqrt(port_var)
        
    def _calculate_sharpe_ratio(self,
                              port_return: float,
                              port_risk: float,
                              risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            port_return: Portfolio return
            port_risk: Portfolio risk
            risk_free: Risk-free rate
            
        Returns:
            float: Sharpe ratio
        """
        return (port_return - risk_free) / port_risk