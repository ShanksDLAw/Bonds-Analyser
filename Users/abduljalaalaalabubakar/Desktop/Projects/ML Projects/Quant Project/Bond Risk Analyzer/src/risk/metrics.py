def get_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            # Get company metrics
            coverage_ratios = self.company.get_coverage_ratios()
            credit_metrics = self.company.get_credit_metrics()
            
            # Get debt analysis
            debt_analysis = self.debt_analyzer.analyze_debt_capacity()
            
            # Calculate core metrics
            default_prob = self._calculate_default_probability(
                coverage_ratios,
                credit_metrics,
                debt_analysis
            )