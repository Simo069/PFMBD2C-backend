from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .services.stats_service import DashboardStatsService

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def dashboard_stats(request):
    data = DashboardStatsService.get_user_stats(request.user)
    return Response(data)
