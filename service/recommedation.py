from pymongo import MongoClient
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity 
from datetime import datetime
from domain.domain import JobFilterRequest, CandidateFilterRequest

class RecommedationService():
    
    def __init__(self):
        self.path_model = "artifacts/word2vec_model.model"
        self.word2vec_model = Word2Vec.load(self.path_model)
        mongo_uri = "mongodb+srv://tm026575:ansu2003@internovacluster.byibt.mongodb.net/?retryWrites=true&w=majority&appName=internovaCluster"
        self.client = MongoClient(mongo_uri)
        self.db = self.client["internova_profile"]
        self.candidate_collection = self.db["candidate_profile"]
        self.job_collection = self.db["job_profile"]
    
    
    def normalize_token(self, token):
        return token.lower().replace(' ', '_')
    
    
    def compute_vector(self, words_list, model):
        vectors = [model[word] for word in words_list if word in model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
    
    
    def calculate_age(self, dob):
        if isinstance(dob, datetime):
            dob_date = dob  # Already a datetime object
        elif isinstance(dob, str):
            dob_date = datetime.strptime(dob, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            raise ValueError("Invalid type for 'dob'. Must be a string or datetime object.")
    
        today = datetime.now()
        return today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))


    def validate_candidate_details(self, candidate: dict, required_fields: dict):
        """
        Validates candidate details and assigns default values if fields are missing.
    
        Args:
            candidate (dict): Candidate data.
            required_fields (dict): Dictionary of required fields with their default values.
    
        Returns:
            dict: Validated candidate details with default values if missing.
        """
        validated_details = {}
        for field, default_value in required_fields.items():
            if not candidate.get(field):
                validated_details[field] = default_value
            else:
                validated_details[field] = candidate.get(field)
    
        return validated_details
    
    
    def CandidateService(self, request: JobFilterRequest):
        
        job = self.job_collection.find_one({"Job Id": int(request.job_id)})
        
        if not job:
            raise ValueError(f"No Job found with Job ID: {request.job_id}")
        
        required_skills = job.get("skills_list", [])
        
        if not required_skills:
            raise ValueError(f"No Required Skills found with Job ID: {request.job_id}")
            
        
        query = {}
        if request.Country:
            query["country"] = request.Country
        if request.location:
            query["location"] = request.location
        if request.Education:
            query["qualification"] = request.Education
        if request.Gender:
            query["gender"] = request.Gender
            
        candidates = list(self.candidate_collection.find(query))
        
        if request.age is not None:
            candidates = [
                candidate for candidate in candidates
                if self.calculate_age(candidate.get("dob")) <= request.age
            ]
            
        if not candidates:
            raise ValueError(f"No candidate matched")
            
        
        job_vector = self.compute_vector([self.normalize_token(skill) for skill in required_skills], self.word2vec_model.wv)
        matched_candidates = []
        
        
        for candidate in candidates:
            candidate_skills = candidate.get("skills", [])
            if not candidate_skills:
                continue  # Skip candidates with no skills

            candidate_vector = self.compute_vector([self.normalize_token(skill) for skill in candidate_skills], self.word2vec_model.wv)
            similarity = cosine_similarity([job_vector], [candidate_vector])[0][0]

            # Include only _id and can_id in the response
            matched_candidates.append({
                "_id": str(candidate["_id"]),  # Convert ObjectId to string
                "can_id": candidate.get("can_id"),
                "similarity": similarity  # Store similarity for sorting
            })

        # Sort by similarity in descending order
        matched_candidates.sort(key=lambda x: x["similarity"], reverse=True)

        # Keep only the top 10 candidates
        top_candidates = matched_candidates[:10]

        # Remove the similarity field from the output
        for candidate in top_candidates:
            del candidate["similarity"]

        return {
            "matched_candidates_count": len(top_candidates),
            "matched_candidates": top_candidates
        }
        
    def JobServices(self, request: CandidateFilterRequest):
        
        can = self.candidate_collection.find_one({"can_id": str(request.can_id)})
        
        if not can:
            raise ValueError(f"No candidate found with Candidate ID: {request.can_id}")
        
        required_fields = {
            'skills': [],          # Default to an empty list
            'gender': "Male",       # Default to "Male" if gender is not provided
            'experience': 0        # Default to 0 if experience is not provided
        }
        
        # Call the validation function
        candidate_details = self.validate_candidate_details(can, required_fields)

        # Extract details
        skills = candidate_details['skills']
        gender = candidate_details['gender']
        experience = candidate_details['experience']
    
        # Convert skills to lowercase for matching
        candidate_skills = [self.normalize_token(skill) for skill in skills]

        # Compute vector for the candidate's skills
        candidate_vector = self.compute_vector(candidate_skills, self.word2vec_model.wv)

        # Filter jobs based on gender and experience
        jobs = self.job_collection.find({
            "Preference": {"$in": ["both", gender]},  # Filter by gender preference
            "min_experiment": {"$lte": experience}    # Filter by minimum experience required
        })
    
        # List to store matched jobs
        matched_jobs = []

        for job in jobs:
            job_skills = job.get('skills_list', [])
            job_skills_lower = [self.normalize_token(skill) for skill in job_skills]
        
            # Compute vector for the job's required skills
            job_vector = self.compute_vector(job_skills_lower, self.word2vec_model.wv)

            # Compute cosine similarity between the job and the candidate vectors
            similarity = cosine_similarity([job_vector], [candidate_vector])[0][0]

            # Add job to the matched list if similarity is above a certain threshold (e.g., 0.5)
            if similarity > 0:
                matched_jobs.append({
                    "_id": str(job["_id"]),  # Return the job document _id
                    "job_id": job.get("Job Id"),  # Return the job's Job Id
                    "similarity": similarity  # Store similarity for sorting
                })

        # Sort by similarity in descending order
        matched_jobs.sort(key=lambda x: x["similarity"], reverse=True)

        # Keep only the top 10 jobs
        top_jobs = matched_jobs[:10]

        # Remove the similarity field from the output
        for job in top_jobs:
            del job["similarity"]

        return {
            "matched_jobs_count": len(top_jobs),
            "matched_jobs": top_jobs
        }
        