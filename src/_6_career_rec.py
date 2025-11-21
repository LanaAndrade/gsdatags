"""
Sistema Inteligente de Recomendação de Carreira.
Versão sem rules.py e sem hybrid_recommendation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import joblib


class CareerRecommender:
    """Recomenda trilhas de carreira a partir de skills e perfil."""

    def __init__(self, model_path: str = "../models/best_pipeline.joblib"):
        """Inicializa o sistema de recomendação de carreira."""
        self.model_path = model_path
        self.model = None
        self.model_loaded = False

        try:
            self.model = joblib.load(model_path)
            self.model_loaded = True
            print("Modelo carregado com sucesso.")
        except Exception as e:
            self.model = None
            self.model_loaded = False
            print(f"Erro ao carregar modelo: {e}")
            print("Usando fallback simples (carreira padrão).")

        self.career_descriptions = {
            "backend": {
                "name": "Backend Developer",
                "description": "Desenvolvimento de servidores, APIs e lógica de negócio.",
                "skills": ["Python", "Java", "Node.js", "SQL", "Docker", "AWS", "Spring", "Go", "REST APIs"],
                "demand": "Alta",
                "salary_range": "R$ 5.000 - R$ 15.000",
            },
            "frontend": {
                "name": "Frontend Developer",
                "description": "Interface do usuário, experiência visual e interatividade.",
                "skills": ["JavaScript", "TypeScript", "React", "Vue", "CSS", "HTML5", "Angular", "SASS"],
                "demand": "Alta",
                "salary_range": "R$ 4.000 - R$ 12.000",
            },
            "data_ml": {
                "name": "Data Scientist / ML Engineer",
                "description": "Análise de dados, machine learning e business intelligence.",
                "skills": ["Python", "SQL", "Pandas", "TensorFlow", "Statistics", "ML", "PyTorch", "Data Visualization"],
                "demand": "Muito Alta",
                "salary_range": "R$ 6.000 - R$ 20.000",
            },
            "devops_sre": {
                "name": "DevOps / SRE Engineer",
                "description": "Infraestrutura, deploy, monitoramento e cloud.",
                "skills": ["Docker", "Kubernetes", "AWS", "Terraform", "CI/CD", "Linux", "Monitoring", "Ansible"],
                "demand": "Alta",
                "salary_range": "R$ 6.000 - R$ 18.000",
            },
            "mobile": {
                "name": "Mobile Developer",
                "description": "Desenvolvimento de aplicativos para iOS e Android.",
                "skills": ["Swift", "Kotlin", "React Native", "Flutter", "iOS", "Android", "Mobile UI/UX"],
                "demand": "Média",
                "salary_range": "R$ 4.000 - R$ 12.000",
            },
            "qa": {
                "name": "QA / Test Engineer",
                "description": "Garantia de qualidade, testes automatizados e CI/CD.",
                "skills": ["Selenium", "Cypress", "Testing", "Automation", "CI/CD", "Jest", "Manual Testing"],
                "demand": "Média",
                "salary_range": "R$ 3.000 - R$ 10.000",
            },
            "ux": {
                "name": "UX/UI Designer",
                "description": "Design de experiência do usuário e pesquisa com usuários.",
                "skills": ["Figma", "User Research", "Wireframing", "Prototyping", "UI/UX", "User Testing"],
                "demand": "Alta",
                "salary_range": "R$ 4.000 - R$ 12.000",
            },
        }

    def _fallback_career(self, skills: List[str]) -> str:
        """Define uma carreira padrão quando o modelo não está disponível."""
        s = {k.lower() for k in skills}
        if {"pandas", "tensorflow", "pytorch", "ml", "machine learning", "data"} & s:
            return "data_ml"
        return "backend"

    def recommend(
        self,
        skills: List[str],
        experience: float = 0.0,
        education: str = "",
        employment: str = "",
        interests: Optional[List[str]] = None,
    ) -> Dict:
        """Recomenda uma carreira com explicações e informações extras."""
        if interests is None:
            interests = []

        if self.model_loaded:
            try:
                input_data = pd.DataFrame(
                    [
                        {
                            "langs_list": skills,
                            "dbs_list": [],
                            "YearsCodePro_num": experience or 0.0,
                            "EdLevel": education or "",
                            "Employment": employment or "",
                        }
                    ]
                )

                ml_prediction = self.model.predict(input_data)[0]

                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(input_data)[0]
                    probability = float(np.max(probabilities))
                else:
                    probability = 0.7

                final_prediction = ml_prediction
                method = "ml"
            except Exception as e:
                print(f"Erro no modelo ML: {e}")
                final_prediction = self._fallback_career(skills)
                probability = 0.6
                method = "fallback"
        else:
            final_prediction = self._fallback_career(skills)
            probability = 0.6
            method = "fallback"

        career_info = self.career_descriptions.get(final_prediction, {})
        explanation = self._generate_explanation(final_prediction, skills, probability, method)

        return {
            "recommended_career": final_prediction,
            "career_name": career_info.get("name", final_prediction),
            "confidence": probability,
            "description": career_info.get("description", "Carreira em tecnologia."),
            "explanation": explanation,
            "suggested_skills": self._get_suggested_skills(final_prediction, skills),
            "demand": career_info.get("demand", "Média"),
            "salary_range": career_info.get("salary_range", "Não especificado"),
            "method_used": method,
            "all_career_options": self._get_alternative_careers(skills, final_prediction),
        }

    def _generate_explanation(
        self,
        career: str,
        user_skills: List[str],
        confidence: float,
        method: str,
    ) -> str:
        """Gera explicação em texto simples baseada nas skills."""
        skill_aliases = {
            "python": ["data_ml", "backend"],
            "javascript": ["frontend", "backend"],
            "java": ["backend"],
            "sql": ["data_ml", "backend"],
            "docker": ["devops_sre", "backend"],
            "react": ["frontend", "mobile"],
            "typescript": ["frontend", "backend"],
            "html": ["frontend"],
            "css": ["frontend"],
            "aws": ["devops_sre", "backend"],
            "tensorflow": ["data_ml"],
            "pandas": ["data_ml"],
        }

        relevant_skills = []
        for skill in user_skills:
            skill_lower = skill.lower()
            if skill_lower in skill_aliases and career in skill_aliases[skill_lower]:
                relevant_skills.append(skill)

        method_text = {
            "ml": "nossa inteligência artificial",
            "fallback": "nosso sistema de recomendação padrão",
        }.get(method, "nosso sistema")

        display_career = self.career_descriptions.get(career, {}).get("name", career)

        if relevant_skills:
            skills_text = ", ".join(relevant_skills[:3])
            return (
                f"{method_text.capitalize()} identificou que sua experiência com "
                f"{skills_text} é muito valorizada em {display_career}. "
                f"Confiança: {confidence:.1%}"
            )
        else:
            return (
                f"{method_text.capitalize()} analisou seu perfil completo e "
                f"recomenda {display_career} como a melhor opção. "
                f"Confiança: {confidence:.1%}"
            )

    def _get_suggested_skills(self, career: str, current_skills: List[str]) -> List[str]:
        """Sugere skills para desenvolvimento na carreira recomendada."""
        career_info = self.career_descriptions.get(career, {})
        career_skills = career_info.get("skills", [])

        current_set = set(s.lower() for s in current_skills)
        suggested = [s for s in career_skills if s.lower() not in current_set]

        return suggested[:4]

    def _get_alternative_careers(self, skills: List[str], main_career: str) -> List[Dict]:
        """Retorna carreiras alternativas baseadas nas skills."""
        alternatives = []
        current_set = set(s.lower() for s in skills)

        for career, info in self.career_descriptions.items():
            if career == main_career:
                continue

            career_skills = set(s.lower() for s in info.get("skills", []))
            common_skills = current_set.intersection(career_skills)
            match_score = len(common_skills) / len(info.get("skills", [])) if info.get("skills") else 0

            if match_score > 0.2:
                alternatives.append(
                    {
                        "career": career,
                        "name": info.get("name", career),
                        "match_score": round(match_score, 2),
                        "common_skills": list(common_skills)[:3],
                    }
                )

        return sorted(alternatives, key=lambda x: x["match_score"], reverse=True)[:3]


def print_recommendation_example(model_path: str):
    """Imprime um exemplo simples de recomendação de carreira no console."""
    recommender = CareerRecommender(model_path)

    skills = ["Python", "SQL", "JavaScript", "Docker"]
    experience = 2.5
    education = "Bachelor's degree"
    employment = "Full-time"
    interests = ["data_science", "cloud"]

    result = recommender.recommend(
        skills=skills,
        experience=experience,
        education=education,
        employment=employment,
        interests=interests,
    )

    print("Recomendação de carreira")
    print(f"Carreira: {result['career_name']}")
    print(f"Confiança: {result['confidence']:.1%}")
    print(f"Descrição: {result['description']}")
    print(f"Explicação: {result['explanation']}")
    print(f"Demanda: {result['demand']}")
    print(f"Faixa salarial: {result['salary_range']}")
    print(f"Skills sugeridas: {', '.join(result['suggested_skills'])}")

    if result["all_career_options"]:
        print("Carreiras alternativas:")
        for alt in result["all_career_options"]:
            print(f"- {alt['name']} (Match: {alt['match_score']:.0%})")
