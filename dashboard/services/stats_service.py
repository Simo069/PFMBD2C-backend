from django.db.models import Count
from django.db.models.functions import TruncDate
from django.utils import timezone
from datetime import timedelta
from chat.models import ChatSession, Message
from document.models import PDFFile

class DashboardStatsService:

    @staticmethod
    def get_user_stats(user):
        """
        Retourne les statistiques du dashboard pour un utilisateur donné
        """
        total_messages = Message.objects.filter(user=user).count()

        sessions = ChatSession.objects.filter(user=user)
        total_sessions = sessions.count()

        total_pdfs = PDFFile.objects.filter(user=user).count()

        # -------------------------
        # Temps moyen par session
        # -------------------------
        total_time = 0
        counted_sessions = 0
        for session in sessions:
            messages = session.messages.order_by("created_at")
            if messages.count() >= 2:
                start = messages.first().created_at
                end = messages.last().created_at
                total_time += (end - start).total_seconds()
                counted_sessions += 1
        avg_session_time = int(total_time / counted_sessions) if counted_sessions > 0 else 0

        # -------------------------
        # Most frequent questions
        # -------------------------
        most_frequent_questions_qs = (
            Message.objects
            .filter(user=user, role='user')
            .values('content')
            .annotate(count=Count('id'))
            .filter(count__gt=1)  
            .order_by('-count')[:5]  
        )
        most_frequent_questions = [q["content"] for q in most_frequent_questions_qs]

        # -------------------------
        # Most used PDFs
        # -------------------------
        pdf_usage = {}
        for session in sessions:
            for pdf_id in session.pdf_ids:
                pdf_usage[pdf_id] = pdf_usage.get(pdf_id, 0) + 1

        pdf_objects = PDFFile.objects.filter(id__in=pdf_usage.keys())
        most_used_pdfs = [
            {
                "pdf_name": pdf.original_filename,
                "usage_count": pdf_usage.get(pdf.id, 0) 
            }
            for pdf in pdf_objects
        ]
        most_used_pdfs = sorted(most_used_pdfs, key=lambda x: x["usage_count"], reverse=True)[:5]

      
        today = timezone.now().date()
        week_ago = today - timedelta(days=6)  
        
        messages_by_day = (
            Message.objects
            .filter(user=user, created_at__date__gte=week_ago, created_at__date__lte=today)
            .annotate(day=TruncDate('created_at'))
            .values('day')
            .annotate(count=Count('id'))
            .order_by('day')
        )
        
        # Récupérer les sessions groupées par jour
        sessions_by_day = (
            ChatSession.objects
            .filter(user=user, created_at__date__gte=week_ago, created_at__date__lte=today)
            .annotate(day=TruncDate('created_at'))
            .values('day')
            .annotate(count=Count('id'))
            .order_by('day')
        )
        
        messages_dict = {item['day']: item['count'] for item in messages_by_day}
        sessions_dict = {item['day']: item['count'] for item in sessions_by_day}
        
        daily_activity = []
        day_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
        
        for i in range(7):
            current_date = week_ago + timedelta(days=i)
            day_name = day_names[current_date.weekday()]
            
            daily_activity.append({
                'day': day_name,
                'date': current_date.isoformat(),
                'messages': messages_dict.get(current_date, 0),
                'sessions': sessions_dict.get(current_date, 0)
            })

        return {
            "total_messages": total_messages,
            "total_sessions": total_sessions,
            "total_pdfs": total_pdfs,
            "avg_session_time": avg_session_time,
            "most_frequent_questions": most_frequent_questions,
            "most_used_pdfs": most_used_pdfs,
            "daily_activity": daily_activity, 
        }