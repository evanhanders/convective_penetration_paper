! ***********************************************************************
!
!   Copyright (C) 2010-2019  Bill Paxton & The MESA Team
!
!   this file is part of mesa.
!
!   mesa is free software; you can redistribute it and/or modify
!   it under the terms of the gnu general library public license as published
!   by the free software foundation; either version 2 of the license, or
!   (at your option) any later version.
!
!   mesa is distributed in the hope that it will be useful, 
!   but without any warranty; without even the implied warranty of
!   merchantability or fitness for a particular purpose.  see the
!   gnu library general public license for more details.
!
!   you should have received a copy of the gnu library general public license
!   along with this software; if not, write to the free software
!   foundation, inc., 59 temple place, suite 330, boston, ma 02111-1307 usa
!
! ***********************************************************************
 
      module run_star_extras

      use star_lib
      use star_def
      use const_def
      use math_lib
      
      implicit none


      integer, parameter :: max_nz = 10000
      real(dp) :: penetration_fraction(max_nz)

      ! these routines are called by the standard run_star check_model
      contains
      

      subroutine extras_controls(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         
         ! this is the place to set any procedure pointers you want to change
         ! e.g., other_wind, other_mixing, other_energy  (see star_data.inc)


         ! the extras functions in this file will not be called
         ! unless you set their function pointers as done below.
         ! otherwise we use a null_ version which does nothing (except warn).

         s% extras_startup => extras_startup
         s% extras_start_step => extras_start_step
         s% extras_check_model => extras_check_model
         s% extras_finish_step => extras_finish_step
         s% extras_after_evolve => extras_after_evolve
         s% how_many_extra_history_columns => how_many_extra_history_columns
         s% data_for_extra_history_columns => data_for_extra_history_columns
         s% how_many_extra_profile_columns => how_many_extra_profile_columns
         s% data_for_extra_profile_columns => data_for_extra_profile_columns  

         s% how_many_extra_history_header_items => how_many_extra_history_header_items
         s% data_for_extra_history_header_items => data_for_extra_history_header_items
         s% how_many_extra_profile_header_items => how_many_extra_profile_header_items
         s% data_for_extra_profile_header_items => data_for_extra_profile_header_items

         s% other_mlt => other_mlt
         s% other_D_mix => other_D_mix

      end subroutine extras_controls

      subroutine other_mlt( &
            id, k, cgrav, m, mstar, r, L, X, &            
            T_face, rho_face, P_face, &
            chiRho_face, chiT_face, &
            Cp_face, opacity_face, grada_face, &            
            alfa, beta, & ! f_face = alfa*f_00 + beta*f_m1
            T_00, T_m1, rho_00, rho_m1, P_00, P_m1, &
            chiRho_for_partials_00, chiT_for_partials_00, &
            chiRho_for_partials_m1, chiT_for_partials_m1, &
            chiRho_00, d_chiRho_00_dlnd, d_chiRho_00_dlnT, &
            chiRho_m1, d_chiRho_m1_dlnd, d_chiRho_m1_dlnT, &
            chiT_00, d_chiT_00_dlnd, d_chiT_00_dlnT, &
            chiT_m1, d_chiT_m1_dlnd, d_chiT_m1_dlnT, &
            Cp_00, d_Cp_00_dlnd, d_Cp_00_dlnT, &
            Cp_m1, d_Cp_m1_dlnd, d_Cp_m1_dlnT, &
            opacity_00, d_opacity_00_dlnd, d_opacity_00_dlnT, &
            opacity_m1, d_opacity_m1_dlnd, d_opacity_m1_dlnT, &
            grada_00, d_grada_00_dlnd, d_grada_00_dlnT, &
            grada_m1, d_grada_m1_dlnd, d_grada_m1_dlnT, &            
            gradr_factor, d_gradr_factor_dw, gradL_composition_term, &
            alpha_semiconvection, semiconvection_option, &
            thermohaline_coeff, thermohaline_option, &
            dominant_iso_for_thermohaline, &
            mixing_length_alpha, alt_scale_height, remove_small_D_limit, &
            MLT_option, Henyey_y_param, Henyey_nu_param, &
            normal_mlt_gradT_factor, &
            prev_conv_vel, max_conv_vel, g_theta, dt, tau, just_gradr, &
            mixing_type, mlt_basics, mlt_partials1, ierr)
         
! UNCOMMENT THIS
         !use star_lib, only: star_mlt_eval
         use eos_lib, only: Radiation_Pressure
         use utils_lib
         
         integer, intent(in) :: id ! id for star         
         integer, intent(in) :: k ! cell number or 0 if not for a particular cell         
         real(dp), intent(in) :: &
            cgrav, m, mstar, r, L, X, &            
            T_face, rho_face, P_face, &
            chiRho_face, chiT_face, &
            Cp_face, opacity_face, grada_face, &            
            alfa, beta, &
            T_00, T_m1, rho_00, rho_m1, P_00, P_m1, &
            chiRho_for_partials_00, chiT_for_partials_00, &
            chiRho_for_partials_m1, chiT_for_partials_m1, &
            chiRho_00, d_chiRho_00_dlnd, d_chiRho_00_dlnT, &
            chiRho_m1, d_chiRho_m1_dlnd, d_chiRho_m1_dlnT, &
            chiT_00, d_chiT_00_dlnd, d_chiT_00_dlnT, &
            chiT_m1, d_chiT_m1_dlnd, d_chiT_m1_dlnT, &
            Cp_00, d_Cp_00_dlnd, d_Cp_00_dlnT, &
            Cp_m1, d_Cp_m1_dlnd, d_Cp_m1_dlnT, &
            opacity_00, d_opacity_00_dlnd, d_opacity_00_dlnT, &
            opacity_m1, d_opacity_m1_dlnd, d_opacity_m1_dlnT, &
            grada_00, d_grada_00_dlnd, d_grada_00_dlnT, &
            grada_m1, d_grada_m1_dlnd, d_grada_m1_dlnT, &
            gradr_factor, d_gradr_factor_dw, gradL_composition_term, &
            alpha_semiconvection, thermohaline_coeff, mixing_length_alpha, &
            Henyey_y_param, Henyey_nu_param, &
            prev_conv_vel, max_conv_vel, g_theta, dt, tau, remove_small_D_limit, &
            normal_mlt_gradT_factor
         logical, intent(in) :: alt_scale_height
         character (len=*), intent(in) :: thermohaline_option, MLT_option, semiconvection_option
         integer, intent(in) :: dominant_iso_for_thermohaline
         logical, intent(in) :: just_gradr
         integer, intent(out) :: mixing_type
         real(dp), intent(inout) :: mlt_basics(:) ! (num_mlt_results)
         real(dp), intent(inout), pointer :: mlt_partials1(:) ! =(num_mlt_partials, num_mlt_results)
         integer, intent(out) :: ierr
         
         type (star_info), pointer :: s
         real(dp), pointer :: mlt_partials(:,:)
         real(dp) :: mlt_partials2(num_mlt_partials, num_mlt_results)
         real(dp) :: p
         integer :: j
         
         mlt_partials(1:num_mlt_partials,1:num_mlt_results) => &
            mlt_partials1(1:num_mlt_partials*num_mlt_results)

         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         
         call star_mlt_eval(  &
            id, k, cgrav, m, mstar, r, L, X, &            
            T_face, rho_face, P_face, &
            chiRho_face, chiT_face, &
            Cp_face, opacity_face, grada_face, &            
            alfa, beta, & ! f_face = alfa*f_00 + beta*f_m1
            T_00, T_m1, rho_00, rho_m1, P_00, P_m1, &
            chiRho_for_partials_00, chiT_for_partials_00, &
            chiRho_for_partials_m1, chiT_for_partials_m1, &
            chiRho_00, d_chiRho_00_dlnd, d_chiRho_00_dlnT, &
            chiRho_m1, d_chiRho_m1_dlnd, d_chiRho_m1_dlnT, &
            chiT_00, d_chiT_00_dlnd, d_chiT_00_dlnT, &
            chiT_m1, d_chiT_m1_dlnd, d_chiT_m1_dlnT, &
            Cp_00, d_Cp_00_dlnd, d_Cp_00_dlnT, &
            Cp_m1, d_Cp_m1_dlnd, d_Cp_m1_dlnT, &
            opacity_00, d_opacity_00_dlnd, d_opacity_00_dlnT, &
            opacity_m1, d_opacity_m1_dlnd, d_opacity_m1_dlnT, &
            grada_00, d_grada_00_dlnd, d_grada_00_dlnT, &
            grada_m1, d_grada_m1_dlnd, d_grada_m1_dlnT, &            
            gradr_factor, d_gradr_factor_dw, gradL_composition_term, &
            alpha_semiconvection, semiconvection_option, &
            thermohaline_coeff, thermohaline_option, &
            dominant_iso_for_thermohaline, &
            mixing_length_alpha, alt_scale_height, remove_small_D_limit, &
            MLT_option, Henyey_y_param, Henyey_nu_param, &
            normal_mlt_gradT_factor, &
            prev_conv_vel, max_conv_vel, g_theta, dt, tau, just_gradr, &
            mixing_type, mlt_basics, mlt_partials1, ierr)

         if (k /= 0) then
            if (is_bad(penetration_fraction(k))) penetration_fraction(k) = 0d0

            p = min(penetration_fraction(k), 0.9d0)

            mlt_basics(mlt_gradT) = mlt_basics(mlt_gradT) * (1d0 - p) + p * mlt_basics(mlt_gradL)
            mlt_partials2 = mlt_partials
            mlt_partials(:,mlt_gradT) = mlt_partials2(:,mlt_gradT) * (1d0 - p) + p * mlt_partials2(:,mlt_gradL)
         else
            return
         end if
      end subroutine other_mlt
      
      
      subroutine other_D_mix(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         integer :: k
         type (star_info), pointer :: s
         ierr = 0

         ! Setup
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         do k=1,s%nz
            if (penetration_fraction(k) > 1d-2) then ! Avoid mixing way too far from the boundary.
               s%D_mix(k) = s%D_mix(k) + penetration_fraction(k) * pow(s%L(k) / (4d0 * pi * pow2(s%r(k)) * s%rho(k)), 1d0/3d0) * s%scale_height(k)
            end if
         end do

      end subroutine other_D_mix

      subroutine extras_startup(id, restart, ierr)
         integer, intent(in) :: id
         logical, intent(in) :: restart
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         penetration_fraction = 0d0
      end subroutine extras_startup
      

      integer function extras_start_step(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s

         ! Setup
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_start_step = 0
         penetration_fraction(1:max_nz) = 0d0

         call update_pen_frac(s, s%gradr, s%gradL)
      
      end function extras_start_step

      subroutine single_boundary(s, k, direction, dgrad)
         type (star_info), pointer :: s
         integer :: k, direction
         real(dp), dimension(:), intent(in) :: dgrad
         real(dp), parameter :: f = 0.86d0
         real(dp), parameter :: xi = 0.6d0
         integer :: j
         real(dp) :: Lint, delta_r, Lavg, RHS
         real(dp) :: dr

         ! direction is the direction of the radiative zone
         if (direction == 1) then
            delta_r = 0d0
            Lint = 0d0

            ! Integrate over CZ
            do j=k,1,-1
!               if (abs(s%lnP(j) - s%lnP(k)) > 2d0) then
                  ! Means we've gone for more than a scale height
!                  exit
!               end if

               if (dgrad(j) < 0d0) then
                  ! Means we've hit a radiative zone
                  exit
               end if

               dr = s%dm(j) / (4d0 * pi * pow2(s%r(j)) * s%rho(j))
               Lint = Lint + s%L_conv(j) * dr
               delta_r = delta_r + dr

!               penetration_fraction(j) = penetration_fraction(j) + 1d0
            end do 
            write(*,*) direction,k,j,delta_r/s%r(1),Lint

            ! Calculate target RHS
            RHS = (1d0 - f) * Lint
            Lavg = Lint / delta_r
            Lint = 0d0

            ! Integrate over RZ until we find the edge of the PZ
            delta_r = 0d0
            do j=k+1,s%nz
               dr = s%dm(j) / (4d0 * pi * pow2(s%r(j)) * s%rho(j))
               delta_r = delta_r + dr
               Lint = Lint + (xi * f * Lavg + s%L(j) * (s%grada(j) / s%gradr(j) - 1d0)) * dr

               if (Lint > RHS) then
                  write(*,*) direction,k,j,Lint,RHS,delta_r/s%scale_height(j),s%grada(j) / s%gradr(j)
                  exit
               end if

               penetration_fraction(j) = penetration_fraction(j) + 1d0
            end do             
         else
            delta_r = 0d0
            Lint = 0d0

            ! Integrate over CZ
            do j=k+1,s%nz
!               if (abs(s%lnP(j) - s%lnP(k)) > 2d0) then
                  ! Means we've gone for more than a scale height
!                  exit
!               end if

               if (dgrad(j) < 0d0) then
                  ! Means we've hit a radiative zone
                  exit
               end if

               dr = s%dm(j) / (4d0 * pi * pow2(s%r(j)) * s%rho(j))
               Lint = Lint + s%L_conv(j) * dr
               delta_r = delta_r + dr

!               penetration_fraction(j) = penetration_fraction(j) + 1d0
            end do 
            write(*,*) direction,k,j,delta_r/s%r(1),Lint

            ! Calculate target RHS
            RHS = (1d0 - f) * Lint
            Lavg = Lint / delta_r
            Lint = 0d0

            ! Integrate over RZ until we find the edge of the PZ
            delta_r = 0d0
            do j=k,1,-1
               dr = s%dm(j) / (4d0 * pi * pow2(s%r(j)) * s%rho(j))
               delta_r = delta_r + dr
               Lint = Lint + (xi * f * Lavg + s%L(j) * (s%grada(j) / s%gradr(j) - 1d0)) * dr

               if (Lint > RHS) then
                  write(*,*) direction,k,j,Lint,RHS,delta_r/s%scale_height(j),s%grada(j) / s%gradr(j)
                  exit
               end if

               penetration_fraction(j) = penetration_fraction(j) + 1d0
            end do                     
         end if

      end subroutine single_boundary

      subroutine update_pen_frac(s, gradR, gradL)
         type (star_info), pointer :: s
         real(dp), dimension(:), intent(in) :: gradR, gradL
         real(dp), dimension(:), allocatable :: dgrad
         real(dp) :: dgradM, dgradP
         real(dp) :: dist, dr
         real(dp) :: cz_height
         integer :: k
         logical :: convective
         ! gradr and gradL are defined on faces.
         ! So conv/rad is defined on faces.
         ! So convective boundaries are defined on cells.
         ! Ignore boundaries on cells 1/nz for now (not clear that that would be meaningful).

         penetration_fraction = 0d0

         allocate(dgrad(s%nz))
         do k=1,s%nz
            dgrad(k) = gradR(k) - gradL(k)
         end do

         do k=2,s%nz-1
            ! Cell k is bounded by faces k,k+1.
            dgradM = dgrad(k)
            dgradP = dgrad(k+1)
            if (dgradM > 0d0 .and. dgradP < 0d0) then
               ! convective at k and radiative at k+1
               call single_boundary(s, k, 1, dgrad)
            end if
            if (dgradP > 0d0 .and. dgradM < 0d0) then
               ! radiative at k and convective at k+1
               call single_boundary(s, k, -1, dgrad)
            end if
         end do

         ! Force it to lie in [0,1].
         do k=1,s%nz
            if (penetration_fraction(k) < 0d0) then
               penetration_fraction(k) = 0d0
            else if (penetration_fraction(k) > 1d0) then
               penetration_fraction(k) = 1d0
            end if
         end do


      end subroutine update_pen_frac

      ! returns either keep_going, retry, or terminate.
      integer function extras_check_model(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_check_model = keep_going         

         ! if you want to check multiple conditions, it can be useful
         ! to set a different termination code depending on which
         ! condition was triggered.  MESA provides 9 customizeable
         ! termination codes, named t_xtra1 .. t_xtra9.  You can
         ! customize the messages that will be printed upon exit by
         ! setting the corresponding termination_code_str value.
         ! termination_code_str(t_xtra1) = 'my termination condition'

         ! by default, indicate where (in the code) MESA terminated
         if (extras_check_model == terminate) s% termination_code = t_extras_check_model
      end function extras_check_model


      integer function how_many_extra_history_columns(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_columns = 0
      end function how_many_extra_history_columns
      
      
      subroutine data_for_extra_history_columns(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         
         ! note: do NOT add the extras names to history_columns.list
         ! the history_columns.list is only for the built-in history column options.
         ! it must not include the new column names you are adding here.
         

      end subroutine data_for_extra_history_columns

      
      integer function how_many_extra_profile_columns(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_profile_columns = 1
      end function how_many_extra_profile_columns
      
      
      subroutine data_for_extra_profile_columns(id, n, nz, names, vals, ierr)
         integer, intent(in) :: id, n, nz
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(nz,n)
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         integer :: k
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return

         names(1) = 'penetration_fraction'
         do k=1,nz
            vals(k,1) = penetration_fraction(k)
         end do
         
         ! note: do NOT add the extra names to profile_columns.list
         ! the profile_columns.list is only for the built-in profile column options.
         ! it must not include the new column names you are adding here.

         ! here is an example for adding a profile column
         !if (n /= 1) stop 'data_for_extra_profile_columns'
         !names(1) = 'beta'
         !do k = 1, nz
         !   vals(k,1) = s% Pgas(k)/s% P(k)
         !end do
         
      end subroutine data_for_extra_profile_columns


      integer function how_many_extra_history_header_items(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_history_header_items = 0
      end function how_many_extra_history_header_items


      subroutine data_for_extra_history_header_items(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_history_column_name) :: names(n)
         real(dp) :: vals(n)
         type(star_info), pointer :: s
         integer, intent(out) :: ierr
         ierr = 0
         call star_ptr(id,s,ierr)
         if(ierr/=0) return

         ! here is an example for adding an extra history header item
         ! also set how_many_extra_history_header_items
         ! names(1) = 'mixing_length_alpha'
         ! vals(1) = s% mixing_length_alpha

      end subroutine data_for_extra_history_header_items


      integer function how_many_extra_profile_header_items(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         how_many_extra_profile_header_items = 0
      end function how_many_extra_profile_header_items


      subroutine data_for_extra_profile_header_items(id, n, names, vals, ierr)
         integer, intent(in) :: id, n
         character (len=maxlen_profile_column_name) :: names(n)
         real(dp) :: vals(n)
         type(star_info), pointer :: s
         integer, intent(out) :: ierr
         ierr = 0
         call star_ptr(id,s,ierr)
         if(ierr/=0) return

         ! here is an example for adding an extra profile header item
         ! also set how_many_extra_profile_header_items
         ! names(1) = 'mixing_length_alpha'
         ! vals(1) = s% mixing_length_alpha

      end subroutine data_for_extra_profile_header_items


      ! returns either keep_going or terminate.
      ! note: cannot request retry; extras_check_model can do that.
      integer function extras_finish_step(id)
         integer, intent(in) :: id
         integer :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
         extras_finish_step = keep_going

         ! to save a profile, 
            ! s% need_to_save_profiles_now = .true.
         ! to update the star log,
            ! s% need_to_update_history_now = .true.

         ! see extras_check_model for information about custom termination codes
         ! by default, indicate where (in the code) MESA terminated
         if (extras_finish_step == terminate) s% termination_code = t_extras_finish_step
      end function extras_finish_step
      
      
      subroutine extras_after_evolve(id, ierr)
         integer, intent(in) :: id
         integer, intent(out) :: ierr
         type (star_info), pointer :: s
         ierr = 0
         call star_ptr(id, s, ierr)
         if (ierr /= 0) return
      end subroutine extras_after_evolve

      end module run_star_extras
      
